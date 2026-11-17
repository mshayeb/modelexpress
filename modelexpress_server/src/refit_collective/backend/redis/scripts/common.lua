-- Helpers shared by the collective group scripts. Redis EVAL has no `require`,
-- so redis.rs prepends this chunk to each script that uses it when the script
-- is built. Only key_range reads KEYS directly; registration_key derives
-- worker registration keys from `worker_key_prefix` passed via ARGV, which
-- works in Redis Cluster EVAL because KEYS[1] (`mx:{refit}c:...`) and
-- `worker_key_prefix` (`mx:{refit}:worker:`) share the `{refit}` hash tag and
-- map to the same slot.

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
end

local function registration_key(worker_key_prefix, worker_id)
  return worker_key_prefix .. worker_id
end

local function registration_matches(worker_id, role, model_name, worker_key_prefix, key)
  key = key or registration_key(worker_key_prefix, worker_id)
  if redis.call('EXISTS', key) ~= 1 then
    return false
  end
  local expected_role = role == 'TRAINER' and '1' or '2'
  return redis.call('HGET', key, 'worker_id') == worker_id
    and redis.call('HGET', key, 'role') == expected_role
    and redis.call('HGET', key, 'model_name') == model_name
end

-- Map newline-separated operation IDs 1-to-1 onto declared KEYS entries.
local function build_op_key_map(op_ids_blob, op_keys)
  local map = {}
  if not op_ids_blob or op_ids_blob == '' then
    return map
  end
  local idx = 1
  for op_id in string.gmatch(op_ids_blob .. '\n', '([^\n]*)\n') do
    if op_id ~= '' and op_keys[idx] then
      map[op_id] = op_keys[idx]
      idx = idx + 1
    end
  end
  return map
end

-- Verify every active operation in ops_set_key was declared in KEYS before any
-- mutation executes, so Redis Cluster checks slot locality upfront.
local function active_ops_covered(ops_set_key, op_key_map)
  local op_ids = redis.call('SMEMBERS', ops_set_key)
  for i = 1, #op_ids do
    if not op_key_map[op_ids[i]] then
      return false
    end
  end
  return true
end

-- A group churn that invalidates the current epoch's in-flight communicator
-- also strands any transfer op still riding it. Abort every non-terminal op
-- this group knows about before the epoch moves out from under it.
local function abort_stale_ops(ops_set_key, op_key_map)
  local op_ids = redis.call('SMEMBERS', ops_set_key)
  for i = 1, #op_ids do
    local op_key = op_key_map[op_ids[i]]
    if op_key then
      local op_state = redis.call('HGET', op_key, 'state')
      if op_state == 'PENDING' or op_state == 'RUNNING' then
        redis.call('HSET', op_key, 'state', 'ABORTED',
          'failure_message', 'group epoch advanced before the transfer completed')
      end
      redis.call('SREM', ops_set_key, op_ids[i])
    end
  end
end

-- Whether any lane in lane_keys has published a bootstrap for this epoch.
local function any_lane_bootstrapped(lane_keys, epoch)
  for i = 1, #lane_keys do
    local bootstrap_epoch = redis.call('HGET', lane_keys[i], 'bootstrap_epoch')
    if bootstrap_epoch and tonumber(bootstrap_epoch) == epoch then
      return true
    end
  end
  return false
end

-- Only a membership that could already have communicator state needs fencing:
-- a READY group, or one where some lane has published a current-epoch
-- bootstrap. In a clean FORMING epoch a membership change just acknowledges
-- the epoch it is in.
local function needs_epoch_bump(group_key, lane_keys, epoch)
  return redis.call('HGET', group_key, 'state') == 'READY'
    or any_lane_bootstrapped(lane_keys, epoch)
end

-- READY means every exact slot/rank is live and has acknowledged this epoch,
-- every lane is bootstrapped for it, and every digest agrees. model_name nil
-- skips the registration check, for a caller that has just swept lapsed
-- registrations out of the participants hash itself.
local function evaluate_readiness(
    group_key, participants_key, digests_key, lane_keys, expected, epoch, model_name, worker_key_prefix)
  local admitted = redis.call('HLEN', participants_key)
  if expected == nil or admitted ~= expected then
    return false
  end

  local records = redis.call('HVALS', participants_key)
  for i = 1, #records do
    local worker_id, role, _, joined_epoch = parse_participant(records[i])
    if not worker_id or tonumber(joined_epoch) ~= epoch then
      return false
    end
    if model_name ~= nil and not registration_matches(worker_id, role, model_name, worker_key_prefix) then
      return false
    end
  end

  for i = 1, #lane_keys do
    local stamped = redis.call('HGET', lane_keys[i], 'bootstrap_epoch')
    if not stamped or tonumber(stamped) ~= epoch then
      return false
    end
  end

  local digest = redis.call('HGET', group_key, 'plan_digest')
  local reported = redis.call('HVALS', digests_key)
  if #reported ~= expected then
    return false
  end
  for i = 1, #reported do
    if reported[i] ~= digest then
      return false
    end
  end
  return true
end

-- Collect KEYS[first..last] into a list.
local function key_range(first, last)
  local keys = {}
  for i = first, last do
    keys[#keys + 1] = KEYS[i]
  end
  return keys
end
