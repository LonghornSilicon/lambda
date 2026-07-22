# CI setup — GitHub Actions + self-hosted runner

The CI workflow at `.github/workflows/ci.yml` has four jobs:

| Job | Runs on | Wall-clock |
|---|---|---|
| `rtl-functional-verification` | GitHub Ubuntu (free) | ~1 min |
| `rtl-synthesis` | GitHub Ubuntu (free) | ~30 s |
| `openlane-sky130` | **self-hosted (your cloud box)** | ~5-10 min |
| `paper-build` | GitHub Ubuntu (free) | ~1 min |

The three light jobs run on GitHub-hosted runners (unlimited for public
repos, free for private up to 2000 min/month). The Sky130 OpenLane job
runs on your always-on cloud server because (a) it needs Docker, which
GitHub-hosted runners support but with image-pull overhead every run,
and (b) you save GitHub Actions minutes for the future heavier jobs
(KV cache, token importance unit, full-chip integration).

## One-time runner setup

### What you need on the cloud server

- **OS**: Ubuntu 22.04+ or similar Linux distro (the runner is glibc-2.31+).
- **Disk**: 30 GB free (Docker images, PDK cache, build artifacts).
- **Network**: outbound HTTPS to `github.com`, `ghcr.io`, `pypi.org`,
  `download.docker.com`.
- **Pre-installed software**:
  - `docker` (in the user's group so it runs without sudo)
  - `python3` with `pip` (Python 3.10+)
  - `librelane` (`pip install --user librelane`)
  - `git`

Verify with:

```sh
docker run --rm hello-world
python3 --version
pip show librelane
```

### Register the runner with GitHub

1. Go to https://github.com/LonghornSilicon/attention-compute-unit/settings/actions/runners
2. Click **New self-hosted runner** → choose **Linux** / **x64**.
3. GitHub gives you a small shell script. It looks like:

```sh
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.X.X.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.X.X/actions-runner-linux-x64-2.X.X.tar.gz
tar xzf actions-runner-linux-x64-2.X.X.tar.gz
./config.sh --url https://github.com/LonghornSilicon/attention-compute-unit \
            --token <ONE-TIME-TOKEN>
```

Run those commands on the cloud server. When `config.sh` prompts:
- Runner name: pick something like `longhorn-cloud-1`.
- Labels: accept defaults (`self-hosted`, `Linux`, `X64`) — the workflow
  uses these.
- Work folder: accept default `_work`.

### Run the runner as a service (so it survives reboots)

```sh
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

The runner should now show as **Idle** under
Settings → Actions → Runners on GitHub. Trigger a CI run (push a
commit, or hit "Run workflow" on the Actions tab) and watch the
`openlane-sky130` job pick up on your runner.

## Verifying the workflow

After registering the runner, push any small change to `master` (or
manually click "Run workflow" on the Actions tab). All four jobs
should run:

- The three GitHub-hosted jobs start within ~30 s.
- The self-hosted Sky130 job picks up on your runner — first run pulls
  the LibreLane Docker image (~6 GB, one-time), subsequent runs are
  fast.

Each job uploads its artifacts (synth log, PDF, GDS+metrics+layout PNG)
so you can download them from the run page on GitHub.

## Extending to future LonghornSilicon components

The bottom of `ci.yml` has commented-out stubs for the rest of the chip:

- **KV Cache Engine** — ChannelQuant compress / decompress pipeline + on-die
  SRAM controller. Verification will look similar to the
  precision controller TBs: a directed TB + a replay TB driving real
  KV traces.
- **Token Importance Unit** — accumulator array tracking attention
  weight per cached token. TB drives synthetic attention rows;
  reference is a simple Python accumulator.
- **Memory Hierarchy Controller** — on-die SRAM / off-chip LPDDR5X
  routing logic (direct, no eDRAM tier). TB needs a workload trace (sequence of accesses)
  and a reference cache model.
- **Full-chip integration** — composes the above + the ACU (precision
  controller). Runs as a `needs:`-gated job that fires only when all
  component jobs pass.

Each new component repeats the same template: a verification job and an
OpenLane job. Once a component lands in the repo, just uncomment the
stub at the bottom of `ci.yml`, point it at the new RTL/OpenLane
config, and push.

## Cost / efficiency

- **GitHub-hosted runners** (Linux): free for public repos, 2000
  min/month free for private. Our three light jobs use ~3 min/run.
- **Self-hosted runner**: pay only for your cloud server uptime
  (which you're already paying). No GitHub Actions minutes consumed
  for the OpenLane job.
- **Caching**: the LibreLane Docker image and Sky130 PDK cache persist
  across runs on the self-hosted runner, so each Sky130 run is ~3 min
  after the first pull.

## Troubleshooting

**Self-hosted runner shows "Offline"** — service crashed or got logged
out. SSH in and run `sudo ./svc.sh status` to see what happened. If
it's wedged, `sudo ./svc.sh stop && sudo ./svc.sh start`.

**OpenLane job fails with "container not found"** — your Docker login
session expired. Run `docker logout && docker login ghcr.io` on the
runner with a GitHub personal access token that has `read:packages`.
This is rare; usually you don't need to log in to pull the librelane
image because it's public.

**Out of disk on the runner** — clean up old `runs/` directories with
`docker system prune -af` and removing
`~/actions-runner/_work/attention-compute-unit/openlane/precision_controller/runs/`.
Cron-job this if it becomes routine.

**FF count assertion fails in `rtl-synthesis`** — someone changed the
RTL parameters (BLOCK_M/BLOCK_N/SCORE_WIDTH) without updating the
expected count in the workflow. Update the workflow's `FF_COUNT`
check to match the new closed-form value.
