# CI setup — GitHub Actions

The CI workflow at `.github/workflows/ci.yml` has four jobs:

| Job | Runs on | Wall-clock |
|---|---|---|
| `rtl-functional-verification` | GitHub Ubuntu (free) | ~1 min |
| `rtl-synthesis` | GitHub Ubuntu (free) | ~30 s |
| `openlane-sky130` | GitHub Ubuntu (free) | ~5-10 min |
| `reference-model` | GitHub Ubuntu (free) | ~1 min |

All four jobs run on GitHub-hosted runners. The OpenLane job pulls the
LibreLane Docker image (~6 GB) and Sky130 PDK on every run — this adds
~3 min of overhead but requires zero infrastructure. For faster
iteration, you can optionally move the OpenLane job to a self-hosted
runner (see below).

## Verifying the workflow

Push any small change to `master` (or manually click "Run workflow"
on the Actions tab). All four jobs should run:

- The three light jobs start within ~30 s.
- The OpenLane job takes ~5-10 min (image pull + synthesis + PnR).

Each job uploads its artifacts (synth log, GDS, metrics, layout PNG)
so you can download them from the run page on GitHub.

## Optional: self-hosted runner for faster OpenLane

If the 5-10 min GitHub-hosted OpenLane runtime becomes a bottleneck,
move it to a self-hosted runner where the Docker image and PDK cache
persist between runs (~3 min runtime after first pull).

### What you need on the server

- **OS**: Ubuntu 22.04+ or similar Linux distro (glibc-2.31+).
- **Architecture**: x86_64 (the LibreLane Docker image is x86_64 only).
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

The runner should be registered at the **organization level**
(`LonghornSilicon`) so every block repo shares the same hardware:

1. Go to https://github.com/organizations/LonghornSilicon/settings/actions/runners
2. Click **New self-hosted runner** → choose **Linux** / **x64**.
3. GitHub gives you a small shell script:

```sh
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.X.X.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.X.X/actions-runner-linux-x64-2.X.X.tar.gz
tar xzf actions-runner-linux-x64-2.X.X.tar.gz
./config.sh --url https://github.com/LonghornSilicon \
            --token <ORG-TOKEN> \
            --labels self-hosted,linux,x64 \
            --unattended
```

### Run as a service

```sh
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

The runner should now show as **Idle** under
Settings → Actions → Runners on GitHub.

### Update the workflow

Change the OpenLane job's `runs-on` from `ubuntu-latest` to
`[self-hosted, linux, x64]`:

```yaml
  openlane-sky130:
    runs-on: [self-hosted, linux, x64]
```

## Extending to other LonghornSilicon blocks

When the Token Importance Unit or Memory Hierarchy Controller repos
come online, each gets the same CI shape:

1. `*-functional-verification` (block-specific testbench)
2. `*-synthesis` (block-specific FF-count assertion)
3. `openlane-sky130` (block-specific OpenLane config, same gate logic)
4. `reference-model` (block-specific `make test-all`)

If the runner is at org-level, the same machine picks up all blocks'
heavy jobs without any per-repo registration.

## Cost / efficiency

- **GitHub-hosted runners** (Linux): free for public repos, 2000
  min/month free for private. Our four jobs use ~10 min/run total.
- **Self-hosted runner** (optional): pay only for your cloud server
  uptime. No GitHub Actions minutes consumed for the OpenLane job.
- **Caching**: on a self-hosted runner, the LibreLane Docker image
  and Sky130 PDK cache persist across runs, cutting OpenLane runtime
  to ~3 min.

## Troubleshooting

**Self-hosted runner shows "Offline"** — service crashed or got logged
out. SSH in and run `sudo ./svc.sh status` to see what happened. If
it's wedged, `sudo ./svc.sh stop && sudo ./svc.sh start`.

**OpenLane job fails with "container not found"** — Docker login
session expired. Run `docker logout && docker login ghcr.io` on the
runner. Usually not needed since the librelane image is public.

**Out of disk on the runner** — clean up old `runs/` directories with
`docker system prune -af` and removing old OpenLane run outputs.

**Yosys synthesis scope** — the full ChannelQuant datapath synthesizes cleanly
(`kv_cache_engine.sv` + the `extra-rtl-sources` list: `cq_key_path`, `cq_value_path`,
`cq_units_syn`, `amax_unit`, `residual_buffer`, `scale_bank`, `sram_controller`).
The behavioral `real`-math oracle (`cq_units.sv` / `cq_fp_pkg.sv`) is TB-only and
is not read by the synth/formal gates.

**Architecture mismatch** — the LibreLane Docker image is x86_64 only.
ARM64 runners cannot run the OpenLane job. Use GitHub-hosted
`ubuntu-latest` (x86_64) or an x86_64 self-hosted runner.
