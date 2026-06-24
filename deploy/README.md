# Deployment

The system is split so the frontend is always-on and free, while the live backend
(which costs compute) is optional and on-demand. When the backend is off, the
frontend automatically plays back a recorded run of real telemetry, so the
public link always shows the system working, at zero cost.

```
  Browser ──HTTPS──> Vercel (Next.js dashboard, always-on, free)
                        │  /api/stats  (server-side proxy)
                        ├──► live:   GCP VM ──HTTP──> engine :9200 (full stack)
                        │            static IP 34.72.85.177
                        └──► replay: bundled recording, looped (backend offline)
```

## Recorded-demo mode (always-on, $0, no backend needed)

`/api/stats` tries the live backend first; if it's unreachable it serves a bundled
recording (`dashboard/src/app/api/stats/replay.json`) of a real run, looped and
advanced by wall-clock. Replayed snapshots carry `_replay: true` and the UI shows
a "Recorded demo" badge with a `?` that explains it: honest, real measured data,
no compute. So the VM is **optional**: start it only for a live demo, and you can
delete it entirely when credits run out, and the showcase keeps working.

There are two bundled recordings:
- `replay.json`: a normal paced run (served in Live mode when the VM is off)
- `replay_maxcap.json`: the fleet under heavy concurrent load (served in Max mode)

Re-record the paced run (with the VM running), from the dashboard folder:
```
cd dashboard
BACKEND=http://34.72.85.177:9200 SECONDS=150 node record-replay.mjs
```

Re-record the max-capacity run: on the VM, drive concurrent load at the servers
and capture while it runs (the engine alone can't load the fleet, it's paced and
sequential):
```
# on the VM, in ~ :
for p in b c d a; do sudo docker exec -d company-$p ./build/bench_async <peer>:50051 2 32 230; done
python3 capture_mc.py        # writes ~/replay_maxcap.json (150 frames)
```
Then pull both files into `dashboard/src/app/api/stats/`, trim each frame's events
to ~30 to keep the bundle small, and `vercel deploy --prod --yes`.

Why the proxy: the browser only ever talks HTTPS to Vercel; Vercel's server
fetches the HTTP backend. No mixed-content block, no CORS, no certs needed.

## Live URLs

- **Frontend (public):** https://rtb-control-plane.vercel.app
- **Backend engine:** http://34.72.85.177:9200/stats  (static IP, only up when the VM runs)

When the backend is stopped, the dashboard shows a "backend sleeping" state and
auto-connects within ~1s of the engine coming back.

## Start the backend (for a demo)

```
gcloud compute instances start rtb-backend --zone=us-central1-a --project=project-9d1cab9d-6622-4e91-aed
```
The containers have `--restart unless-stopped`, so the 4 bidders + engine + the
retrainer come back automatically (~30-90s after boot). Same static IP every time,
so the Vercel frontend reconnects with no changes. Account: `mazaaqure@gmail.com`.

Only you can start/stop the VM; a viewer can't. But a viewer never needs you to:
with the VM off they still see the full dashboard on recorded data. Start the VM
only when you personally want to show real-time data (e.g. in an interview).

## Stop the backend (save credits)

```
gcloud compute instances stop rtb-backend --zone=us-central1-a --project=project-9d1cab9d-6622-4e91-aed
```
Stopped = no compute charge. You still pay for the 30GB disk and the reserved
static IP while the VM exists. To wipe it entirely (truly $0):
```
gcloud compute instances delete rtb-backend --zone=us-central1-a --project=project-9d1cab9d-6622-4e91-aed
gcloud compute addresses delete rtb-backend-ip --region=us-central1 --project=project-9d1cab9d-6622-4e91-aed
```

## Cost

- VM (e2-standard-8, 8 vCPU) while running: ~$0.27/hr (~$6/day if left on).
- Stopped: ~$5-7/month (30GB pd-balanced disk + the reserved-but-idle static IP).
- Deleted: $0. Frontend on Vercel: free, always.
- Check remaining free-trial/promo credits in the Cloud Console: Billing ->
  account `01DE16-B41BEB-4F413D` -> Credits (the balance isn't available via CLI).
- **Long-term portfolio:** keep the VM stopped (or deleted) and let viewers see the
  recorded demo. The Vercel link stays up for free indefinitely; live mode is just
  a bonus you turn on for demos.

## Redeploy the frontend (after dashboard changes)

```
cd dashboard
vercel deploy --prod --yes
```
The backend URL is baked into `src/app/api/stats/route.ts` (the Vercel default);
override per-environment with a `BACKEND_URL` env var if the IP ever changes.

## Rebuild the backend (after engine/bidder changes)

The VM has the source in `~` and runs the stack with docker compose (project
`moham`, network `moham_default`). To ship an engine change: scp the changed files
into `~/go-engine/...`, then on the VM:
```
cd ~ && sudo docker compose build engine && sudo docker compose up -d engine
```
The compose build runs `go test ./...` first. Recreating just `engine` leaves the
4 bidders + retrainer running. Note: `~/rebuild-engine.sh` runs the engine on the
wrong network (`rtbnet`); use the compose commands above instead, which keep it
on `moham_default` where the bidders are reachable.
