# TODO

## Repository layout

- [ ] After all active experiments and queued jobs finish, move the MALOQ Git
  checkout from
  `/dataset/seongsu/shared-home/workspace/project` to
  `/dataset/seongsu/shared-home/workspace/project/MALOQ`.
  - Do not move it while live jobs use
    `PYTHONPATH=/dataset/seongsu/shared-home/workspace/project/src`.
  - Update absolute project paths in launchers, queue configuration,
    automation, documentation, and tests.
  - Validate the Git remote, imports, experiment launchers, queue workers, and
    GPU monitor from the new location before starting new runs.
