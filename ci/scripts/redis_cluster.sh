#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Manage a 6-node (3 primaries + 3 replicas) local Redis Cluster inside a single
# Docker container for CI and local integration testing.
#
# Usage:
#   ci/scripts/redis_cluster.sh start
#   ci/scripts/redis_cluster.sh kill-primary [port]
#   ci/scripts/redis_cluster.sh stop

set -euo pipefail

CONTAINER="${MX_REDIS_CLUSTER_CONTAINER:-mx-redis-cluster-test}"
IMAGE="${MX_REDIS_CLUSTER_IMAGE:-redis:7-alpine}"
BASE_PORT="${MX_REDIS_CLUSTER_BASE_PORT:-7000}"
END_PORT=$((BASE_PORT + 5))
BUS_BASE_PORT=$((BASE_PORT + 10000))
BUS_END_PORT=$((END_PORT + 10000))

usage() {
  echo "Usage: $0 {start|kill-primary [port]|stop}" >&2
  exit 1
}

container_running() {
  docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -qx 'true'
}

wait_node_ping() {
  local port="$1"
  local attempt=0
  while [ "${attempt}" -lt 50 ]; do
    if docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${port}" PING 2>/dev/null | grep -qx 'PONG'; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 0.1
  done
  echo "Redis node on port ${port} did not respond to PING in time" >&2
  return 1
}

cmd_start() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

  docker run -d --rm \
    --name "${CONTAINER}" \
    -p "127.0.0.1:${BASE_PORT}-${END_PORT}:${BASE_PORT}-${END_PORT}" \
    -p "127.0.0.1:${BUS_BASE_PORT}-${BUS_END_PORT}:${BUS_BASE_PORT}-${BUS_END_PORT}" \
    "${IMAGE}" sh -c 'trap "exit 0" TERM INT; while :; do sleep 1; done' >/dev/null

  local addrs=""
  for offset in 0 1 2 3 4 5; do
    local port=$((BASE_PORT + offset))
    local bus_port=$((port + 10000))
    docker exec "${CONTAINER}" sh -c \
      "mkdir -p /tmp/node-${port} && redis-server \
        --daemonize yes \
        --bind 0.0.0.0 \
        --protected-mode no \
        --port ${port} \
        --cluster-enabled yes \
        --cluster-config-file /tmp/node-${port}/nodes.conf \
        --cluster-node-timeout 2000 \
        --cluster-announce-ip 127.0.0.1 \
        --cluster-announce-port ${port} \
        --cluster-announce-bus-port ${bus_port} \
        --dir /tmp/node-${port} \
        --appendonly no \
        --save '' \
        --pidfile /tmp/node-${port}/redis.pid \
        --logfile /tmp/node-${port}/redis.log"
    wait_node_ping "${port}"
    addrs="${addrs} 127.0.0.1:${port}"
  done

  # shellcheck disable=SC2086
  docker exec "${CONTAINER}" redis-cli --cluster create ${addrs} --cluster-replicas 1 --cluster-yes >/dev/null

  local ready=0
  local attempt=0
  while [ "${attempt}" -lt 60 ]; do
    ready=1
    for offset in 0 1 2 3 4 5; do
      local port=$((BASE_PORT + offset))
      local info
      info="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${port}" CLUSTER INFO 2>/dev/null || true)"
      if ! printf '%s\n' "${info}" | grep -q 'cluster_state:ok' || \
         ! printf '%s\n' "${info}" | grep -q 'cluster_slots_assigned:16384'; then
        ready=0
        break
      fi
    done
    if [ "${ready}" -eq 1 ]; then
      break
    fi
    attempt=$((attempt + 1))
    sleep 0.25
  done

  if [ "${ready}" -ne 1 ]; then
    echo "Redis Cluster failed to converge within timeout" >&2
    docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${BASE_PORT}" CLUSTER INFO >&2 || true
    exit 1
  fi

  local p0="${BASE_PORT}"
  local p1=$((BASE_PORT + 1))
  local p2=$((BASE_PORT + 2))
  echo "redis://127.0.0.1:${p0},redis://127.0.0.1:${p1},redis://127.0.0.1:${p2}"
}

find_surviving_port() {
  local excluded="$1"
  for offset in 0 1 2 3 4 5; do
    local port=$((BASE_PORT + offset))
    if [ "${port}" -eq "${excluded}" ]; then
      continue
    fi
    if docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${port}" PING 2>/dev/null | grep -qx 'PONG'; then
      echo "${port}"
      return 0
    fi
  done
  return 1
}

cmd_kill_primary() {
  if ! container_running; then
    echo "Redis Cluster container '${CONTAINER}' is not running" >&2
    exit 1
  fi

  local query_port=""
  query_port="$(find_surviving_port 0)" || {
    echo "No reachable Redis Cluster nodes in '${CONTAINER}'" >&2
    exit 1
  }

  local cluster_nodes
  cluster_nodes="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${query_port}" CLUSTER NODES)"

  local target_port="${1:-}"
  if [ -z "${target_port}" ]; then
    target_port="$(printf '%s\n' "${cluster_nodes}" | \
      awk '$3 ~ /master/ && $3 !~ /fail/ && $9 != "" { split($2, hp, "@"); n = split(hp[1], p, ":"); print p[n]; exit }')"
  fi

  if [ -z "${target_port}" ]; then
    echo "Could not locate an active primary node to kill" >&2
    exit 1
  fi

  local target_node_id replica_port
  target_node_id="$(printf '%s\n' "${cluster_nodes}" | \
    awk -v tp="${target_port}" '$3 ~ /master/ && $3 !~ /fail/ { split($2, hp, "@"); n = split(hp[1], p, ":"); if (p[n] == tp) { print $1; exit } }')"
  if [ -n "${target_node_id}" ]; then
    replica_port="$(printf '%s\n' "${cluster_nodes}" | \
      awk -v mid="${target_node_id}" '$3 ~ /slave/ && $3 !~ /fail/ && $4 == mid { split($2, hp, "@"); n = split(hp[1], p, ":"); print p[n]; exit }')"
  else
    replica_port=""
  fi

  local master_offset
  master_offset="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${target_port}" INFO replication 2>/dev/null | \
    tr -d '\r' | awk -F: '$1 == "master_repl_offset" { print $2; exit }')"
  if [ -z "${master_offset}" ]; then
    echo "Could not determine master_repl_offset on primary port ${target_port}" >&2
    exit 1
  fi

  local synced=0
  local rep_attempt=0
  while [ "${rep_attempt}" -lt 50 ]; do
    if [ -n "${replica_port}" ]; then
      local rep_info link_status replica_offset
      rep_info="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${replica_port}" INFO replication 2>/dev/null | tr -d '\r' || true)"
      link_status="$(printf '%s\n' "${rep_info}" | awk -F: '$1 == "master_link_status" { print $2; exit }')"
      replica_offset="$(printf '%s\n' "${rep_info}" | awk -F: '$1 == "slave_repl_offset" { print $2; exit }')"
      if [ "${link_status}" = "up" ] && [ -n "${replica_offset}" ] && [ "${replica_offset}" -ge "${master_offset}" ]; then
        synced=1
        break
      fi
    else
      local primary_info slave_offset
      primary_info="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${target_port}" INFO replication 2>/dev/null | tr -d '\r' || true)"
      slave_offset="$(printf '%s\n' "${primary_info}" | awk -F'[,=]' '/^slave[0-9]+:.*state=online/ { for (i = 1; i <= NF; i++) if ($i == "offset") { print $(i + 1); exit } }')"
      if [ -n "${slave_offset}" ] && [ "${slave_offset}" -ge "${master_offset}" ]; then
        synced=1
        break
      fi
    fi
    rep_attempt=$((rep_attempt + 1))
    sleep 0.1
  done

  if [ "${synced}" -ne 1 ]; then
    echo "Replica for primary port ${target_port} did not reach replication offset ${master_offset} within timeout" >&2
    if [ -n "${replica_port}" ]; then
      docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${replica_port}" INFO replication >&2 || true
    else
      docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${target_port}" INFO replication >&2 || true
    fi
    exit 1
  fi

  docker exec "${CONTAINER}" sh -c \
    "pid=\$(cat /tmp/node-${target_port}/redis.pid 2>/dev/null || true); if [ -n \"\${pid}\" ]; then kill -9 \"\${pid}\"; fi"

  local survivor
  survivor="$(find_surviving_port "${target_port}")" || {
    echo "No surviving Redis Cluster nodes after killing port ${target_port}" >&2
    exit 1
  }

  local attempt=0
  while [ "${attempt}" -lt 120 ]; do
    local info nodes
    info="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${survivor}" CLUSTER INFO 2>/dev/null || true)"
    nodes="$(docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${survivor}" CLUSTER NODES 2>/dev/null || true)"
    if printf '%s\n' "${info}" | grep -q 'cluster_state:ok' && \
       printf '%s\n' "${info}" | grep -q 'cluster_slots_assigned:16384' && \
       printf '%s\n' "${nodes}" | awk -v tp="${target_port}" 'BEGIN { found=0 } $3 ~ /master/ && $3 !~ /fail/ { split($2, hp, "@"); split(hp[1], p, ":"); if (p[2] == tp) found=1 } END { exit found }'; then
      echo "Killed primary on port ${target_port}; failover complete and cluster_state:ok"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 0.25
  done

  echo "Cluster did not finish failover cleanly after killing port ${target_port}" >&2
  docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${survivor}" CLUSTER INFO >&2 || true
  docker exec "${CONTAINER}" redis-cli -h 127.0.0.1 -p "${survivor}" CLUSTER NODES >&2 || true
  exit 1
}

cmd_stop() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}

case "${1:-}" in
  start)
    cmd_start
    ;;
  kill-primary)
    shift
    cmd_kill_primary "${1:-}"
    ;;
  stop)
    cmd_stop
    ;;
  *)
    usage
    ;;
esac
