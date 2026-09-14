"""Live transports connecting the orchestrator to a running sim or robot process:
shared-memory camera frames, UDP arm/state/gaze/object channels. Counterpart to
sources.py's offline ReplaySource -- same SensorFrame contract, live inputs."""
