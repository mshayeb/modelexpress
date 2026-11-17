// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end Redis Cluster integration and failover tests across all four
//! Redis-backed metadata subsystems (Registry, P2P, Refit, Collective Refit).
//!
//! Run against the local 6-node cluster helper:
//!
//! ```sh
//! ./ci/scripts/redis_cluster.sh start
//! cargo test -p model-express-workspace-tests --test redis_cluster_integration \
//!     -- --include-ignored --test-threads=1
//! ./ci/scripts/redis_cluster.sh stop
//! ```

#![allow(clippy::expect_used, clippy::arithmetic_side_effects)]

use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroU16;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use modelexpress_common::grpc::p2p::{
    BackendFramework, MxSourceType, SourceIdentity, SourceStatus, WorkerMetadata,
    worker_metadata::BackendMetadata,
};
use modelexpress_common::grpc::refit::{
    CreateTrainerMeshRequest, CreateWeightVersionRequest, CreateWeightVersionShardRequest,
    DeleteTrainerMeshRequest, DeleteVersionLeaseRequest, DeleteWeightVersionRequest,
    DeleteWeightVersionShardRequest, GetTrainerShardMetadataRequest,
    GetTrainerShardMetadataResponse, GetWeightVersionShardMetadataRequest,
    GetWeightVersionShardMetadataResponse, ListWeightVersionShardsRequest,
    ListWeightVersionsRequest, RegisterVersionLeaseRequest, RegisterWorkerRequest,
    TrainerTensorsMetadata, UpdateTrainerMeshRequest, WeightPayloadFormat, WeightVersionShard,
    WeightVersionState, WorkerRegistration, WorkerRole,
    refit_service_client::RefitServiceClient,
    refit_worker_service_server::{RefitWorkerService, RefitWorkerServiceServer},
};
use modelexpress_common::grpc::refit_collective::{
    CollectiveGroupSpec, CollectiveGroupState, CollectiveRole, CollectiveTransferState,
    CreateCollectiveTransferRequest, DeleteCollectiveTransferRequest, GetCollectiveGroupRequest,
    GetCollectiveTransferRequest, JoinCollectiveGroupRequest, LaneKind, LaneSpec,
    PublishGroupBootstrapRequest, ReportCollectiveTransferRequest,
    refit_collective_service_client::RefitCollectiveServiceClient,
};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use modelexpress_server::backend_config::BackendConfig;
use modelexpress_server::config::ServerConfig;
use modelexpress_server::p2p::{
    backend::{MetadataBackend, redis::RedisBackend},
    source_identity::compute_mx_source_id,
};
use modelexpress_server::refit::backend::redis::RedisRefitBackend;
use modelexpress_server::refit_collective::backend::redis::RedisCollectiveBackend;
use modelexpress_server::registry::backend::{
    ClaimOutcome, RegistryBackend, redis::RedisRegistryBackend,
};
use modelexpress_server::run_server;
use redis::cluster::ClusterClient;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(1);

const DEFAULT_CLUSTER_SEEDS: &[&str] = &[
    "redis://127.0.0.1:7000",
    "redis://127.0.0.1:7001",
    "redis://127.0.0.1:7002",
    "redis://127.0.0.1:7003",
    "redis://127.0.0.1:7004",
    "redis://127.0.0.1:7005",
];

fn cluster_seed_urls() -> Vec<String> {
    let raw = std::env::var("MX_REDIS_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("REDIS_URL")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });

    match raw {
        Some(list) => list
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect(),
        None => DEFAULT_CLUSTER_SEEDS
            .iter()
            .map(|url| (*url).to_string())
            .collect(),
    }
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn unique_id(tag: &str) -> String {
    let seq = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    format!("mx-cluster-{tag}-{nanos}-{seq}")
}

fn sample_identity(model_name: &str, step: u64, layout_sig: &str) -> SourceIdentity {
    let mut extra_parameters = HashMap::new();
    extra_parameters.insert("training_step".to_string(), step.to_string());
    extra_parameters.insert("layout_signature".to_string(), layout_sig.to_string());
    SourceIdentity {
        mx_version: "0.8.0".to_string(),
        mx_source_type: MxSourceType::Weights as i32,
        model_name: model_name.to_string(),
        backend_framework: BackendFramework::Vllm as i32,
        tensor_parallel_size: 2,
        pipeline_parallel_size: 1,
        expert_parallel_size: 0,
        dtype: "bfloat16".to_string(),
        quantization: String::new(),
        extra_parameters,
        revision: format!("rev-{step}"),
        ..Default::default()
    }
}

fn sample_worker(rank: u32, status: SourceStatus, endpoint_port: u16) -> WorkerMetadata {
    WorkerMetadata {
        worker_rank: rank,
        backend_metadata: Some(BackendMetadata::NixlMetadata(vec![0xAB; 16])),
        status: status as i32,
        updated_at: 1_700_000_000_000 + i64::from(rank),
        metadata_endpoint: format!("127.0.0.1:{endpoint_port}"),
        agent_name: format!("agent-{rank}"),
        worker_grpc_endpoint: format!("127.0.0.1:{}", endpoint_port + 100),
        accelerator: "cuda".to_string(),
        ..Default::default()
    }
}

async fn spawn_mock_standalone_redis() -> (String, oneshot::Sender<()>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock standalone redis");
    let addr = listener.local_addr().expect("mock addr");
    let url = format!("redis://{addr}");
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();

    let handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                accepted = listener.accept() => {
                    let Ok((stream, _)) = accepted else { break };
                    tokio::spawn(handle_mock_resp_connection(stream));
                }
            }
        }
    });

    (url, stop_tx, handle)
}

async fn read_resp_command(
    reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>,
) -> Option<Vec<String>> {
    let mut line = String::new();
    let n = reader.read_line(&mut line).await.ok()?;
    if n == 0 {
        return None;
    }
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if let Some(count_str) = trimmed.strip_prefix('*') {
        let count: usize = count_str.parse().ok()?;
        let mut args = Vec::with_capacity(count);
        for _ in 0..count {
            let mut len_line = String::new();
            if reader.read_line(&mut len_line).await.ok()? == 0 {
                return None;
            }
            let len: isize = len_line
                .trim_end_matches(['\r', '\n'])
                .strip_prefix('$')?
                .parse()
                .ok()?;
            if len < 0 {
                args.push(String::new());
                continue;
            }
            let len_usize = usize::try_from(len).ok()?;
            let mut buf = vec![0u8; len_usize + 2];
            reader.read_exact(&mut buf).await.ok()?;
            buf.truncate(len_usize);
            args.push(String::from_utf8_lossy(&buf).into_owned());
        }
        Some(args)
    } else {
        Some(
            trimmed
                .split_whitespace()
                .map(ToString::to_string)
                .collect(),
        )
    }
}

async fn handle_mock_resp_connection(stream: TcpStream) {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    while let Some(args) = read_resp_command(&mut reader).await {
        let cmd = args
            .first()
            .map(|s| s.to_ascii_uppercase())
            .unwrap_or_default();
        let reply: &[u8] = if cmd == "CLUSTER" {
            b"-ERR This instance has cluster support disabled\r\n"
        } else if cmd == "PING" {
            b"+PONG\r\n"
        } else if cmd == "HELLO" {
            b"-ERR unknown command 'HELLO'\r\n"
        } else if cmd == "INFO" {
            b"$19\r\ncluster_enabled:0\r\n\r\n"
        } else {
            b"+OK\r\n"
        };
        if write_half.write_all(reply).await.is_err() {
            break;
        }
    }
}

struct BoundWorker {
    metadata: Vec<u8>,
}

#[tonic::async_trait]
impl RefitWorkerService for BoundWorker {
    async fn get_trainer_shard_metadata(
        &self,
        _request: tonic::Request<GetTrainerShardMetadataRequest>,
    ) -> Result<tonic::Response<GetTrainerShardMetadataResponse>, tonic::Status> {
        Ok(tonic::Response::new(GetTrainerShardMetadataResponse {
            metadata: self.metadata.clone(),
            metadata_digest: format!("{:x}", Sha256::digest(&self.metadata)),
        }))
    }

    async fn get_weight_version_shard_metadata(
        &self,
        _request: tonic::Request<GetWeightVersionShardMetadataRequest>,
    ) -> Result<tonic::Response<GetWeightVersionShardMetadataResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used in cluster test"))
    }
}

async fn start_bound_worker(
    offset: u64,
) -> (
    TrainerTensorsMetadata,
    oneshot::Sender<()>,
    JoinHandle<ServerResult>,
) {
    let metadata = serde_json::to_vec(&serde_json::json!({
        "tensors": [{
            "name": "weight",
            "dtype": "torch.bfloat16",
            "elsize": 2,
            "full_shape": [8],
            "shards": [{"shard_offset": [offset], "shape": [4]}],
        }]
    }))
    .expect("encode bound coverage");
    let logical_shard_id = format!("{:x}", Sha256::digest(&metadata));
    let port = free_port();
    let (tx, rx) = oneshot::channel();
    let worker = BoundWorker { metadata };
    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RefitWorkerServiceServer::new(worker))
            .serve_with_shutdown(([127, 0, 0, 1], port).into(), async {
                let _ = rx.await;
            })
            .await?;
        Ok(())
    });

    for _ in 0..100 {
        if tonic::transport::Endpoint::new(format!("http://127.0.0.1:{port}"))
            .expect("endpoint")
            .connect()
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    (
        TrainerTensorsMetadata {
            logical_shard_id,
            metadata_endpoint: format!("127.0.0.1:{port}"),
        },
        tx,
        handle,
    )
}

fn start_cluster_server(
    port: u16,
    metrics_port: u16,
    urls: Vec<String>,
) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let mut config = ServerConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = NonZeroU16::new(port).expect("non-zero gRPC port");
    config.server.metrics_port = metrics_port;
    config.cache.eviction.enabled = false;

    let (tx, rx) = oneshot::channel();
    let handle = tokio::spawn(run_server(
        config,
        BackendConfig::RedisCluster { urls },
        async move {
            let _ = rx.await;
        },
    ));
    (tx, handle)
}

async fn connect_grpc(
    port: u16,
) -> (
    RefitServiceClient<Channel>,
    RefitCollectiveServiceClient<Channel>,
) {
    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if let Ok(channel) = Channel::from_shared(endpoint.clone())
            .expect("valid endpoint")
            .connect()
            .await
        {
            return (
                RefitServiceClient::new(channel.clone()),
                RefitCollectiveServiceClient::new(channel),
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server on port {port} never became reachable");
}

async fn stop_task(tx: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    let _ = tx.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("task timed out during shutdown")
        .expect("task panicked")
        .expect("task returned error");
}

async fn scrape_metrics(metrics_port: u16) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", metrics_port))
        .await
        .expect("connect metrics port");
    stream
        .write_all(b"GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await
        .expect("write /metrics request");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("read /metrics response");
    String::from_utf8(response).expect("metrics response is UTF-8")
}

async fn cluster_keyslot(conn: &mut redis::aio::MultiplexedConnection, key: &str) -> u16 {
    redis::cmd("CLUSTER")
        .arg("KEYSLOT")
        .arg(key)
        .query_async(conn)
        .await
        .expect("CLUSTER KEYSLOT")
}

async fn scan_cluster_keys(urls: &[String], pattern: &str) -> Vec<String> {
    let seed = redis::Client::open(urls[0].as_str()).expect("open seed");
    let mut seed_conn = seed
        .get_multiplexed_async_connection()
        .await
        .expect("connect seed");
    let raw_slots: redis::Value = redis::cmd("CLUSTER")
        .arg("SLOTS")
        .query_async(&mut seed_conn)
        .await
        .expect("CLUSTER SLOTS");

    let mut masters = BTreeSet::new();
    if let redis::Value::Array(entries) = raw_slots {
        for entry in entries {
            if let redis::Value::Array(fields) = entry
                && let Some(redis::Value::Array(master_info)) = fields.get(2)
                && let (Some(redis::Value::BulkString(host_bytes)), Some(redis::Value::Int(port))) =
                    (master_info.first(), master_info.get(1))
            {
                let host = String::from_utf8_lossy(host_bytes);
                masters.insert(format!("redis://{host}:{port}"));
            }
        }
    }
    assert!(!masters.is_empty(), "expected at least one cluster primary");

    let mut found = BTreeSet::new();
    for node_url in masters {
        let client = redis::Client::open(node_url.as_str()).expect("open master node");
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .expect("connect master node");
        let mut cursor = 0u64;
        loop {
            let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(pattern)
                .arg("COUNT")
                .arg(200)
                .query_async(&mut conn)
                .await
                .expect("SCAN master");
            found.extend(batch);
            cursor = next_cursor;
            if cursor == 0 {
                break;
            }
        }
    }
    found.into_iter().collect()
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn standalone_backend_rejects_cluster_node_fast() {
    let seed = cluster_seed_urls()
        .into_iter()
        .next()
        .expect("at least one seed URL");
    let timeout = Duration::from_secs(5);

    let registry_err = tokio::time::timeout(timeout, RedisRegistryBackend::new(&seed).connect())
        .await
        .expect("standalone registry topology check timed out")
        .expect_err("standalone registry must reject cluster node")
        .to_string();
    assert!(
        registry_err.contains("cluster") || registry_err.contains("MX_REDIS_MODE"),
        "unexpected registry error: {registry_err}"
    );

    let p2p_err = tokio::time::timeout(timeout, RedisBackend::new(&seed).connect())
        .await
        .expect("standalone p2p topology check timed out")
        .expect_err("standalone p2p must reject cluster node")
        .to_string();
    assert!(
        p2p_err.contains("cluster") || p2p_err.contains("MX_REDIS_MODE"),
        "unexpected p2p error: {p2p_err}"
    );

    let refit_err = tokio::time::timeout(timeout, RedisRefitBackend::connect(&seed))
        .await
        .expect("standalone refit topology check timed out")
        .err()
        .expect("standalone refit must reject cluster node")
        .to_string();
    assert!(
        refit_err.contains("cluster") || refit_err.contains("MX_REDIS_MODE"),
        "unexpected refit error: {refit_err}"
    );

    let collective_err = tokio::time::timeout(timeout, RedisCollectiveBackend::connect(&seed))
        .await
        .expect("standalone collective topology check timed out")
        .err()
        .expect("standalone collective must reject cluster node")
        .to_string();
    assert!(
        collective_err.contains("cluster") || collective_err.contains("MX_REDIS_MODE"),
        "unexpected collective error: {collective_err}"
    );
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn cluster_backend_rejects_standalone_node_fast() {
    let (standalone_url, mock_guard) = match std::env::var("MX_TEST_STANDALONE_REDIS_URL") {
        Ok(url) if !url.trim().is_empty() => (url, None),
        _ => {
            let (url, stop_tx, handle) = spawn_mock_standalone_redis().await;
            (url, Some((stop_tx, handle)))
        }
    };
    let urls = vec![standalone_url];
    let timeout = Duration::from_secs(5);

    let registry_err =
        tokio::time::timeout(timeout, RedisRegistryBackend::new_cluster(&urls).connect())
            .await
            .expect("cluster registry rejection timed out")
            .expect_err("cluster registry must reject standalone node")
            .to_string();
    assert!(
        registry_err.to_lowercase().contains("standalone")
            || registry_err.to_lowercase().contains("cluster"),
        "unexpected cluster registry error: {registry_err}"
    );

    let p2p_err = tokio::time::timeout(timeout, RedisBackend::new_cluster(&urls).connect())
        .await
        .expect("cluster p2p rejection timed out")
        .expect_err("cluster p2p must reject standalone node")
        .to_string();
    assert!(
        p2p_err.to_lowercase().contains("standalone") || p2p_err.to_lowercase().contains("cluster"),
        "unexpected cluster p2p error: {p2p_err}"
    );

    let refit_err = tokio::time::timeout(timeout, RedisRefitBackend::connect_cluster(&urls))
        .await
        .expect("cluster refit rejection timed out")
        .err()
        .expect("cluster refit must reject standalone node")
        .to_string();
    assert!(
        refit_err.to_lowercase().contains("standalone")
            || refit_err.to_lowercase().contains("cluster"),
        "unexpected cluster refit error: {refit_err}"
    );

    let collective_err =
        tokio::time::timeout(timeout, RedisCollectiveBackend::connect_cluster(&urls))
            .await
            .expect("cluster collective rejection timed out")
            .err()
            .expect("cluster collective must reject standalone node")
            .to_string();
    assert!(
        collective_err.to_lowercase().contains("standalone")
            || collective_err.to_lowercase().contains("cluster"),
        "unexpected cluster collective error: {collective_err}"
    );

    if let Some((stop_tx, handle)) = mock_guard {
        let _ = stop_tx.send(());
        let _ = handle.await;
    }
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn single_seed_url_discovers_full_cluster() {
    let single_seed = vec![
        cluster_seed_urls()
            .into_iter()
            .next()
            .expect("at least one seed URL"),
    ];
    let backend = RedisRegistryBackend::new_cluster(&single_seed);
    backend
        .connect()
        .await
        .expect("connect registry with single cluster seed URL");

    let prefix = unique_id("seed-discovery");
    let models: Vec<String> = (0..12).map(|idx| format!("{prefix}/model-{idx}")).collect();

    for model in &models {
        backend
            .set_status(
                model,
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("write model via single seed");
    }

    let all_records = backend
        .get_models_by_last_used(None)
        .await
        .expect("scan models across all discovered primaries");
    let found: BTreeSet<String> = all_records
        .into_iter()
        .map(|record| record.model_name)
        .filter(|name| name.starts_with(&prefix))
        .collect();
    let expected: BTreeSet<String> = models.iter().cloned().collect();
    assert_eq!(
        found, expected,
        "every model across primaries must be discovered"
    );

    for model in &models {
        backend
            .delete_model(model)
            .await
            .expect("cleanup seed-discovery model");
    }
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn registry_multi_slot_crud_lua_leases_and_scan() {
    let urls = cluster_seed_urls();
    let backend = RedisRegistryBackend::new_cluster(&urls);
    backend
        .connect()
        .await
        .expect("connect registry cluster backend");

    let prefix = unique_id("reg");
    let providers = [
        ModelProvider::HuggingFace,
        ModelProvider::Ngc,
        ModelProvider::Gcs,
        ModelProvider::S3,
    ];
    let mut models = Vec::new();
    for idx in 0..14 {
        models.push(format!("{prefix}/org-{idx}/model-{idx}"));
    }
    models.push(format!("{prefix}/tricky/{{brace}}:colon/a"));
    models.push(format!("{prefix}/tricky/multi:{{x}}:{{y}}"));

    for (idx, model) in models.iter().enumerate() {
        let provider = providers[idx % providers.len()];
        backend
            .set_status(
                model,
                provider,
                ModelStatus::DOWNLOADED,
                Some(format!("msg-{idx}")),
            )
            .await
            .expect("set_status");
        assert_eq!(
            backend.get_status(model).await.expect("get_status"),
            Some(ModelStatus::DOWNLOADED)
        );
        let record = backend
            .get_model_record(model)
            .await
            .expect("get_model_record")
            .expect("record must exist");
        assert_eq!(record.model_name, *model);
        assert_eq!(record.provider, provider);
        backend.touch_model(model).await.expect("touch_model");
    }

    // Exercise Lua lease scripts across multi-slot and tricky model names
    let lease_targets = [&models[0], &models[14], &models[15]];
    for target in lease_targets {
        backend
            .delete_model(target)
            .await
            .expect("clear before claim");

        let first_claim = backend
            .try_claim_for_download(
                target,
                ModelProvider::HuggingFace,
                "claim-1",
                Duration::from_secs(1),
            )
            .await
            .expect("initial claim");
        assert_eq!(first_claim, ClaimOutcome::Claimed);

        let contended = backend
            .try_claim_for_download(
                target,
                ModelProvider::HuggingFace,
                "claim-other",
                Duration::from_secs(5),
            )
            .await
            .expect("contended claim");
        assert_eq!(
            contended,
            ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)
        );

        tokio::time::sleep(Duration::from_millis(1150)).await;
        let takeover = backend
            .try_claim_for_download(
                target,
                ModelProvider::HuggingFace,
                "claim-2",
                Duration::from_secs(30),
            )
            .await
            .expect("expired lease takeover");
        assert_eq!(takeover, ClaimOutcome::TookOver);

        assert!(
            backend
                .refresh_download_claim(
                    target,
                    ModelProvider::HuggingFace,
                    "claim-2",
                    Duration::from_secs(30)
                )
                .await
                .expect("refresh owned lease")
        );
        assert!(
            backend
                .finish_download_claim(
                    target,
                    ModelProvider::HuggingFace,
                    "claim-2",
                    ModelStatus::ERROR,
                    Some("simulated error".to_string()),
                )
                .await
                .expect("finish claim with ERROR")
        );
        assert!(
            backend
                .try_reset_error_for_retry(
                    target,
                    ModelProvider::HuggingFace,
                    "claim-3",
                    Duration::from_secs(30)
                )
                .await
                .expect("reset error for retry")
        );
        assert!(
            backend
                .finish_download_claim(
                    target,
                    ModelProvider::HuggingFace,
                    "claim-3",
                    ModelStatus::DOWNLOADED,
                    None,
                )
                .await
                .expect("finish retry with DOWNLOADED")
        );
    }

    // Verify Lua key hash-tagging & multi-slot coverage using raw ClusterClient
    let lease_probe_model = format!("{prefix}/slot-probe:{{{prefix}}}");
    assert_eq!(
        backend
            .try_claim_for_download(
                &lease_probe_model,
                ModelProvider::Ngc,
                "probe-claim",
                Duration::from_secs(30),
            )
            .await
            .expect("claim probe model"),
        ClaimOutcome::Claimed
    );

    let keys = scan_cluster_keys(&urls, &format!("*{prefix}*")).await;
    assert!(keys.len() >= models.len(), "expected scanned cluster keys");

    let seed_client = redis::Client::open(urls[0].as_str()).expect("open seed");
    let mut slot_conn = seed_client
        .get_multiplexed_async_connection()
        .await
        .expect("connect seed for KEYSLOT");
    let cluster_client = ClusterClient::new(urls.clone()).expect("ClusterClient");
    let mut raw_cluster = cluster_client
        .get_async_connection()
        .await
        .expect("raw ClusterConnection");

    let mut distinct_slots = BTreeSet::new();
    for key in &keys {
        distinct_slots.insert(cluster_keyslot(&mut slot_conn, key).await);
        let exists: bool = redis::cmd("EXISTS")
            .arg(key)
            .query_async(&mut raw_cluster)
            .await
            .expect("EXISTS via ClusterClient");
        assert!(exists, "key {key} should exist via ClusterClient");
    }
    for tricky in [&models[14], &models[15]] {
        let base_slot = cluster_keyslot(&mut slot_conn, &format!("mx:model:{{m:{tricky}}}")).await;
        for provider_name in ["HuggingFace", "Ngc", "Gcs", "S3"] {
            assert_eq!(
                cluster_keyslot(
                    &mut slot_conn,
                    &format!("mx:model:{provider_name}:{{m:{tricky}}}")
                )
                .await,
                base_slot,
                "all provider and legacy keys for {tricky} must share one hash slot"
            );
        }
    }
    assert!(
        distinct_slots.len() > 1,
        "expected >=16 models to span multiple hash slots, got {distinct_slots:?}"
    );

    let (downloading, downloaded, error_count) = backend
        .get_status_counts()
        .await
        .expect("get_status_counts");
    assert!(downloading >= 1, "lease_probe_model should be downloading");
    assert!(downloaded >= models.len() as u32);
    let _ = error_count;

    let scanned_models: BTreeSet<String> = backend
        .get_models_by_last_used(None)
        .await
        .expect("get_models_by_last_used")
        .into_iter()
        .map(|record| record.model_name)
        .filter(|name| name.starts_with(&prefix))
        .collect();
    for model in &models {
        assert!(
            scanned_models.contains(model),
            "missing scanned model {model}"
        );
    }
    assert!(scanned_models.contains(&lease_probe_model));

    for model in &models {
        backend.delete_model(model).await.expect("cleanup model");
    }
    backend
        .delete_model(&lease_probe_model)
        .await
        .expect("cleanup probe model");
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn p2p_multi_slot_publish_lua_leases_reap_and_scan() {
    let urls = cluster_seed_urls();
    let backend = RedisBackend::new_cluster(&urls);
    backend
        .connect()
        .await
        .expect("connect p2p cluster backend");

    let run_tag = unique_id("p2p");
    let mut source_entries = Vec::new();
    for idx in 0..12u64 {
        let model_name = format!("{run_tag}/model-{idx}");
        let identity = sample_identity(&model_name, 100 + idx, &format!("sig-{idx}"));
        let source_id = compute_mx_source_id(&identity).expect("compute source_id");
        let worker_a = format!("{run_tag}-worker-{idx}-a");
        let worker_b = format!("{run_tag}-worker-{idx}-b");

        for rank in [0u32, 1u32] {
            backend
                .publish_metadata(
                    &identity,
                    &worker_a,
                    sample_worker(
                        rank,
                        SourceStatus::Ready,
                        18000 + u16::try_from(idx).unwrap_or(0),
                    ),
                    "pod-a",
                    "uid-a",
                    "ns-a",
                )
                .await
                .expect("publish worker_a rank");
        }
        backend
            .publish_metadata(
                &identity,
                &worker_b,
                sample_worker(
                    0,
                    SourceStatus::Ready,
                    19000 + u16::try_from(idx).unwrap_or(0),
                ),
                "pod-b",
                "uid-b",
                "ns-b",
            )
            .await
            .expect("publish worker_b");

        source_entries.push((source_id, model_name, worker_a, worker_b));
    }

    // Check raw hash slots across the 12 source_ids
    let seed = redis::Client::open(urls[0].as_str()).expect("open seed");
    let mut slot_conn = seed
        .get_multiplexed_async_connection()
        .await
        .expect("connect seed");
    let mut slots = BTreeSet::new();
    for (sid, _, _, _) in &source_entries {
        slots.insert(cluster_keyslot(&mut slot_conn, &format!("mx:source:{{{sid}}}")).await);
    }
    assert!(
        slots.len() > 1,
        "expected 12 source_ids across >1 hash slot"
    );

    // Verify get_metadata & Lua rank status patch on one source
    let (first_sid, target_model, worker_a, worker_b) = &source_entries[0];
    let record = backend
        .get_metadata(first_sid, worker_a)
        .await
        .expect("get_metadata")
        .expect("metadata record exists");
    assert_eq!(record.workers.len(), 2);
    assert_eq!(record.workers[0].worker_rank, 0);
    assert_eq!(record.workers[1].worker_rank, 1);

    backend
        .update_status(
            first_sid,
            worker_a,
            0,
            SourceStatus::Initializing,
            1_700_000_999_999,
            Some(0.42),
        )
        .await
        .expect("Lua update_status rank patch");
    let patched = backend
        .get_metadata(first_sid, worker_a)
        .await
        .expect("get_metadata after patch")
        .expect("record exists");
    assert_eq!(patched.workers[0].status, SourceStatus::Initializing as i32);
    assert_eq!(patched.workers[0].updated_at, 1_700_000_999_999);
    backend
        .update_status(
            first_sid,
            worker_a,
            0,
            SourceStatus::Ready,
            1_700_001_000_000,
            Some(0.1),
        )
        .await
        .expect("restore SourceStatus::Ready");

    // Multi-primary SCAN via list_sources & list_workers
    let sources_map: HashMap<String, String> = backend
        .list_sources()
        .await
        .expect("list_sources")
        .into_iter()
        .collect();
    for (sid, model_name, _, _) in &source_entries {
        assert_eq!(sources_map.get(sid), Some(model_name));
    }

    let ready_workers = backend
        .list_workers(None, Some(SourceStatus::Ready))
        .await
        .expect("list_workers Ready");
    for (sid, _, wa, wb) in &source_entries {
        assert!(
            ready_workers
                .iter()
                .any(|w| &w.source_id == sid && &w.worker_id == wa),
            "missing worker {wa}"
        );
        assert!(
            ready_workers
                .iter()
                .any(|w| &w.source_id == sid && &w.worker_id == wb),
            "missing worker {wb}"
        );
    }

    let filtered = backend
        .list_workers_filtered(
            None,
            Some(SourceStatus::Ready),
            Some(target_model.clone()),
            Some(1),
            Some(100),
            Some(1_700_000_000_000),
            Some(5),
        )
        .await
        .expect("list_workers_filtered");
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].worker_id, *worker_a);

    // Donor lease Lua scripts
    let compat_key = format!("{run_tag}-compat-key");
    assert!(
        backend
            .try_acquire_donor_lease(&compat_key, "owner-1", Duration::from_secs(30))
            .await
            .expect("acquire donor lease")
    );
    assert!(
        !backend
            .try_acquire_donor_lease(&compat_key, "owner-2", Duration::from_secs(30))
            .await
            .expect("contend donor lease")
    );
    assert!(
        backend
            .renew_donor_lease(&compat_key, "owner-1", Duration::from_secs(30))
            .await
            .expect("renew donor lease")
    );
    assert!(
        backend
            .release_donor_lease(&compat_key, "owner-1")
            .await
            .expect("release donor lease")
    );
    assert!(
        backend
            .try_acquire_donor_lease(&compat_key, "owner-2", Duration::from_secs(30))
            .await
            .expect("reacquire donor lease")
    );
    assert!(
        backend
            .release_donor_lease(&compat_key, "owner-2")
            .await
            .expect("cleanup donor lease")
    );

    // Worker removal & source removal
    backend
        .remove_worker(first_sid, worker_a)
        .await
        .expect("remove_worker");
    assert!(
        backend
            .get_metadata(first_sid, worker_a)
            .await
            .expect("get removed worker")
            .is_none()
    );
    assert!(
        backend
            .get_metadata(first_sid, worker_b)
            .await
            .expect("sibling worker remains")
            .is_some()
    );
    for (sid, _, _, _) in &source_entries {
        backend.remove_metadata(sid).await.expect("remove_metadata");
    }
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn full_server_refit_collective_and_metrics_over_redis_cluster() {
    let urls = cluster_seed_urls();
    let grpc_port = free_port();
    let metrics_port = free_port();
    let (stop_srv, server_handle) = start_cluster_server(grpc_port, metrics_port, urls);
    let (mut refit, mut collective) = connect_grpc(grpc_port).await;

    // 1. RefitService end-to-end flow with BoundWorker shard binding
    let refit_model = unique_id("refit-model");
    let (meta0, stop_w0, h_w0) = start_bound_worker(0).await;
    let (meta1, stop_w1, h_w1) = start_bound_worker(4).await;
    let trainer0 = unique_id("trainer-0");
    let trainer1 = unique_id("trainer-1");
    let trainer_rep = unique_id("trainer-rep");
    let generator_id = unique_id("generator-0");

    for (wid, ep) in [
        (&trainer0, meta0.metadata_endpoint.clone()),
        (&trainer1, meta1.metadata_endpoint.clone()),
        (&trainer_rep, meta0.metadata_endpoint.clone()),
    ] {
        refit
            .register_worker(RegisterWorkerRequest {
                worker: Some(WorkerRegistration {
                    worker_id: wid.clone(),
                    role: WorkerRole::Trainer.into(),
                    model_name: refit_model.clone(),
                    expires_at_unix_ms: 0,
                    refit_endpoint: ep,
                }),
                ttl_seconds: 60,
            })
            .await
            .expect("register refit trainer");
    }
    refit
        .register_worker(RegisterWorkerRequest {
            worker: Some(WorkerRegistration {
                worker_id: generator_id.clone(),
                role: WorkerRole::Generator.into(),
                model_name: refit_model.clone(),
                expires_at_unix_ms: 0,
                refit_endpoint: String::new(),
            }),
            ttl_seconds: 60,
        })
        .await
        .expect("register refit generator");

    let mesh = refit
        .create_trainer_mesh(CreateTrainerMeshRequest {
            model_name: refit_model.clone(),
            idempotency_key: unique_id("mesh"),
            workers: HashMap::from([
                (trainer0.clone(), meta0.clone()),
                (trainer1.clone(), meta1.clone()),
            ]),
        })
        .await
        .expect("create_trainer_mesh")
        .into_inner()
        .mesh
        .expect("mesh present");

    let version = refit
        .create_weight_version(CreateWeightVersionRequest {
            model_name: refit_model.clone(),
            idempotency_key: unique_id("ver"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            state: WeightVersionState::Staging.into(),
            trainer_mesh_id: Some(mesh.mesh_id.clone()),
            version_number: Some(1),
            ..Default::default()
        })
        .await
        .expect("create_weight_version")
        .into_inner()
        .version
        .expect("version present");

    refit
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(WeightVersionShard {
                version_id: version.uid.clone(),
                logical_shard_id: meta0.logical_shard_id.clone(),
                worker_id: trainer0.clone(),
                stable_metadata_digest: meta0.logical_shard_id.clone(),
                metadata_endpoint: meta0.metadata_endpoint.clone(),
                version_metadata_digest: String::new(),
            }),
        })
        .await
        .expect("publish shard 0");

    let published_resp = refit
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(WeightVersionShard {
                version_id: version.uid.clone(),
                logical_shard_id: meta1.logical_shard_id.clone(),
                worker_id: trainer1.clone(),
                stable_metadata_digest: meta1.logical_shard_id.clone(),
                metadata_endpoint: meta1.metadata_endpoint.clone(),
                version_metadata_digest: String::new(),
            }),
        })
        .await
        .expect("publish shard 1")
        .into_inner();
    assert_eq!(
        published_resp.version.expect("updated version").state,
        WeightVersionState::Ready as i32
    );

    let versions = refit
        .list_weight_versions(ListWeightVersionsRequest {
            model_name: refit_model.clone(),
            trainer_mesh_id: Some(mesh.mesh_id.clone()),
        })
        .await
        .expect("list_weight_versions")
        .into_inner()
        .versions;
    assert_eq!(versions.len(), 1);

    let shards = refit
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: version.uid.clone(),
        })
        .await
        .expect("list_weight_version_shards")
        .into_inner()
        .shards;
    assert_eq!(shards.len(), 2);

    let lease = refit
        .register_version_lease(RegisterVersionLeaseRequest {
            version_id: version.uid.clone(),
            worker_id: generator_id.clone(),
            ttl_seconds: 60,
        })
        .await
        .expect("register_version_lease")
        .into_inner()
        .lease
        .expect("lease present");

    let fenced = refit
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: mesh.generation,
            workers: HashMap::from([
                (trainer_rep.clone(), meta0.clone()),
                (trainer1.clone(), meta1.clone()),
            ]),
        })
        .await
        .expect_err("lease must fence trainer mesh update");
    assert_eq!(fenced.code(), tonic::Code::FailedPrecondition);

    assert!(
        refit
            .delete_version_lease(DeleteVersionLeaseRequest {
                version_id: version.uid.clone(),
                lease_id: lease.lease_id,
                worker_id: generator_id.clone(),
            })
            .await
            .expect("delete_version_lease")
            .into_inner()
            .deleted
    );

    let updated_mesh = refit
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: mesh.generation,
            workers: HashMap::from([
                (trainer_rep.clone(), meta0.clone()),
                (trainer1.clone(), meta1.clone()),
            ]),
        })
        .await
        .expect("update_trainer_mesh after lease deletion")
        .into_inner()
        .mesh
        .expect("updated mesh");
    assert_eq!(updated_mesh.generation, mesh.generation + 1);

    refit
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: version.uid.clone(),
        })
        .await
        .expect("delete_weight_version");
    for (logical_shard_id, worker_id) in [
        (&meta0.logical_shard_id, &trainer0),
        (&meta1.logical_shard_id, &trainer1),
    ] {
        match refit
            .delete_weight_version_shard(DeleteWeightVersionShardRequest {
                version_id: version.uid.clone(),
                logical_shard_id: logical_shard_id.clone(),
                worker_id: worker_id.clone(),
            })
            .await
        {
            Ok(resp) => assert!(resp.into_inner().deleted),
            Err(status) if status.code() == tonic::Code::NotFound => {}
            Err(err) => panic!("delete_weight_version_shard unexpected error: {err}"),
        }
    }
    refit
        .delete_trainer_mesh(DeleteTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
        })
        .await
        .expect("delete_trainer_mesh");

    // 2. RefitCollectiveService end-to-end flow (Lua cluster hash-tag scripts)
    let coll_model = unique_id("coll-model");
    let c_t0 = unique_id("ct0");
    let c_t1 = unique_id("ct1");
    let c_g0 = unique_id("cg0");
    for (wid, role) in [
        (&c_t0, WorkerRole::Trainer),
        (&c_t1, WorkerRole::Trainer),
        (&c_g0, WorkerRole::Generator),
    ] {
        refit
            .register_worker(RegisterWorkerRequest {
                worker: Some(WorkerRegistration {
                    worker_id: wid.clone(),
                    role: role.into(),
                    model_name: coll_model.clone(),
                    expires_at_unix_ms: 0,
                    refit_endpoint: String::new(),
                }),
                ttl_seconds: 60,
            })
            .await
            .expect("register collective worker");
    }

    let trainer_slots = vec!["t0".to_string(), "t1".to_string()];
    let generator_slots = vec!["g0".to_string()];
    let group_spec = CollectiveGroupSpec {
        model_name: coll_model.clone(),
        expected_trainer_slots: trainer_slots.clone(),
        expected_generator_slots: generator_slots.clone(),
        lanes: vec![
            LaneSpec {
                lane_id: 3,
                kind: LaneKind::Reshard.into(),
                trainer_slots: trainer_slots.clone(),
                generator_slots: generator_slots.clone(),
            },
            LaneSpec {
                lane_id: 7,
                kind: LaneKind::Broadcast.into(),
                trainer_slots,
                generator_slots,
            },
        ],
    };

    let m_t0 = collective
        .join_collective_group(JoinCollectiveGroupRequest {
            spec: Some(group_spec.clone()),
            slot_id: "t0".to_string(),
            worker_id: c_t0.clone(),
            role: CollectiveRole::Trainer.into(),
            index_in_role: 0,
            plan_digest: "plan-digest".to_string(),
            plan_source: None,
        })
        .await
        .expect("join t0")
        .into_inner();
    for (slot, wid, role, idx) in [
        ("t1", &c_t1, CollectiveRole::Trainer, 1),
        ("g0", &c_g0, CollectiveRole::Generator, 0),
    ] {
        collective
            .join_collective_group(JoinCollectiveGroupRequest {
                spec: Some(group_spec.clone()),
                slot_id: slot.to_string(),
                worker_id: wid.clone(),
                role: role.into(),
                index_in_role: idx,
                plan_digest: "plan-digest".to_string(),
                plan_source: None,
            })
            .await
            .expect("join participant");
    }

    for lane_id in [3u32, 7u32] {
        collective
            .publish_group_bootstrap(PublishGroupBootstrapRequest {
                group_id: m_t0.group_id.clone(),
                epoch: m_t0.epoch,
                lane_id,
                worker_id: c_t0.clone(),
                nccl_unique_id: vec![7u8; 128],
            })
            .await
            .expect("publish bootstrap");
    }

    let ready_group = collective
        .get_collective_group(GetCollectiveGroupRequest {
            group_id: m_t0.group_id.clone(),
        })
        .await
        .expect("get_collective_group")
        .into_inner();
    assert_eq!(ready_group.state, i32::from(CollectiveGroupState::Ready));

    let transfer = collective
        .create_collective_transfer(CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "coll-version-1".to_string(),
            idempotency_key: unique_id("coll-op"),
            expected_epoch: ready_group.epoch,
        })
        .await
        .expect("create_collective_transfer")
        .into_inner();

    for wid in [&c_t0, &c_t1, &c_g0] {
        collective
            .report_collective_transfer(ReportCollectiveTransferRequest {
                operation_id: transfer.operation_id.clone(),
                group_id: transfer.group_id.clone(),
                epoch: transfer.epoch,
                worker_id: wid.clone(),
                succeeded: true,
                message: String::new(),
            })
            .await
            .expect("report_collective_transfer");
    }

    let completed = collective
        .get_collective_transfer(GetCollectiveTransferRequest {
            operation_id: transfer.operation_id.clone(),
        })
        .await
        .expect("get_collective_transfer")
        .into_inner();
    assert_eq!(
        completed.state,
        i32::from(CollectiveTransferState::Complete)
    );

    collective
        .delete_collective_transfer(DeleteCollectiveTransferRequest {
            operation_id: transfer.operation_id.clone(),
        })
        .await
        .expect("delete_collective_transfer");

    // Verify digest change aborts an in-flight collective transfer with declared KEYS
    let inflight_transfer = collective
        .create_collective_transfer(CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "coll-version-2".to_string(),
            idempotency_key: unique_id("coll-op-abort"),
            expected_epoch: ready_group.epoch,
        })
        .await
        .expect("create second collective transfer")
        .into_inner();
    let bumped = collective
        .join_collective_group(JoinCollectiveGroupRequest {
            spec: Some(group_spec),
            slot_id: "t0".to_string(),
            worker_id: c_t0,
            role: CollectiveRole::Trainer.into(),
            index_in_role: 0,
            plan_digest: "plan-digest-v2".to_string(),
            plan_source: None,
        })
        .await
        .expect("rejoin t0 with new digest")
        .into_inner();
    assert!(bumped.epoch > ready_group.epoch);
    let aborted = collective
        .get_collective_transfer(GetCollectiveTransferRequest {
            operation_id: inflight_transfer.operation_id.clone(),
        })
        .await
        .expect("get aborted collective transfer")
        .into_inner();
    assert_eq!(aborted.state, i32::from(CollectiveTransferState::Aborted));
    collective
        .delete_collective_transfer(DeleteCollectiveTransferRequest {
            operation_id: inflight_transfer.operation_id,
        })
        .await
        .expect("delete aborted collective transfer");

    // 3. Verify Prometheus mx_build_info exports backend="redis-cluster"
    let metrics_body = scrape_metrics(metrics_port).await;
    assert!(
        metrics_body.contains("mx_build_info")
            && metrics_body.contains(r#"backend="redis-cluster""#),
        "expected mx_build_info with backend=\"redis-cluster\", got:\n{metrics_body}"
    );

    stop_task(stop_w0, h_w0).await;
    stop_task(stop_w1, h_w1).await;
    stop_task(stop_srv, server_handle).await;
}

#[tokio::test]
#[ignore = "requires a live Redis Cluster (ci/scripts/redis_cluster.sh start)"]
async fn zz_cluster_survives_primary_failover() {
    let urls = cluster_seed_urls();
    let registry = RedisRegistryBackend::new_cluster(&urls);
    registry
        .connect()
        .await
        .expect("connect registry pre-failover");
    let p2p = RedisBackend::new_cluster(&urls);
    p2p.connect().await.expect("connect p2p pre-failover");

    let run_prefix = unique_id("failover");
    let pre_models: Vec<String> = (0..18)
        .map(|idx| format!("{run_prefix}/pre-model-{idx}"))
        .collect();
    for model in &pre_models {
        registry
            .set_status(
                model,
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("write pre-failover model");
    }

    let mut pre_sources = Vec::new();
    for idx in 0..12u64 {
        let model = format!("{run_prefix}/pre-p2p-{idx}");
        let identity = sample_identity(&model, 50 + idx, &format!("failover-sig-{idx}"));
        let source_id = compute_mx_source_id(&identity).expect("source_id");
        let worker_id = format!("{run_prefix}-worker-{idx}");
        p2p.publish_metadata(
            &identity,
            &worker_id,
            sample_worker(0, SourceStatus::Ready, 17500),
            "pod",
            "uid",
            "ns",
        )
        .await
        .expect("publish pre-failover p2p source");
        pre_sources.push((source_id, model, worker_id));
    }

    if std::env::var("MX_TEST_SKIP_CLUSTER_FAILOVER").as_deref() == Ok("1") {
        for model in &pre_models {
            let _ = registry.delete_model(model).await;
        }
        for (sid, _, _) in &pre_sources {
            let _ = p2p.remove_metadata(sid).await;
        }
        return;
    }

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ci/scripts/redis_cluster.sh");
    let output = Command::new("bash")
        .arg(&script)
        .arg("kill-primary")
        .output()
        .expect("invoke redis_cluster.sh kill-primary");
    assert!(
        output.status.success(),
        "redis_cluster.sh kill-primary failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Existing backend handles must recover topology and read all pre-failover keys
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let reg_res = registry.get_models_by_last_used(None).await;
        let p2p_res = p2p.list_sources().await;
        if let (Ok(records), Ok(sources)) = (reg_res, p2p_res) {
            let found_models: BTreeSet<String> = records
                .into_iter()
                .map(|r| r.model_name)
                .filter(|name| name.starts_with(&run_prefix))
                .collect();
            let source_map: HashMap<String, String> = sources.into_iter().collect();
            let all_models_present = pre_models.iter().all(|m| found_models.contains(m));
            let all_sources_present = pre_sources
                .iter()
                .all(|(sid, model, _)| source_map.get(sid) == Some(model));
            if all_models_present && all_sources_present {
                break;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "existing backend handles did not recover all pre-failover keys within 15s"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // Write post-failover keys across slots and confirm reads + SCAN still cover everything
    let post_models: Vec<String> = (0..6)
        .map(|idx| format!("{run_prefix}/post-model-{idx}"))
        .collect();
    for model in &post_models {
        registry
            .set_status(model, ModelProvider::S3, ModelStatus::DOWNLOADED, None)
            .await
            .expect("write post-failover model");
        assert_eq!(
            registry
                .get_status(model)
                .await
                .expect("read post-failover model"),
            Some(ModelStatus::DOWNLOADED)
        );
    }

    let all_records: BTreeSet<String> = registry
        .get_models_by_last_used(None)
        .await
        .expect("scan post-failover registry")
        .into_iter()
        .map(|r| r.model_name)
        .filter(|name| name.starts_with(&run_prefix))
        .collect();
    for model in pre_models.iter().chain(post_models.iter()) {
        assert!(
            all_records.contains(model),
            "missing {model} after failover"
        );
    }

    for model in pre_models.iter().chain(post_models.iter()) {
        let _ = registry.delete_model(model).await;
    }
    for (sid, _, _) in &pre_sources {
        let _ = p2p.remove_metadata(sid).await;
    }
}
