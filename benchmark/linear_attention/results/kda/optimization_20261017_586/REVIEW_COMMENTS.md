# PR #9 review follow-up — 2026-10-17

Reviewed the comments on performance tip `5218e247`. The runtime fix normalizes
`stream=None` to legacy stream 0 before either backward launch path branches.
The cached kernel and its safe-gate/grouped-head postpasses now receive the same
stream as the cold path. Explicit stream values, public signatures, workspace
ownership and all GPU kernels are unchanged.

## RED/GREEN verification on 586 GPU 1

- New direct `pygraph.execute(..., handle=None)` regression: the old executor
  fails 10 of 12 cases with `int(None)` in safe-gate/head-reduction postpasses;
  two equal-head/non-safe cases pass. Both short/uncut and long/chain inputs
  are exercised, including each grouped-head folding direction.
- The fixed, source-pinned six-file regression suite passes **101 cases** in
  128.77 s, including all 12 new cases, the original execution/replay tests,
  warm binding, dO ownership, elected stores, host ABI and resource tests.
- The 36 mock cold/warm binding combinations additionally cover `None`, 0 and
  17. The new `None` checks fail 24 combinations before the fix; all 36 pass
  afterward. This is source-driven Python ABI coverage, not a GPU measurement.
- Ten local source-only ABI/election checks and the FP64 oracle's CPU self-test
  pass. Changed Python syntax, pre-commit and whitespace checks pass.

The export's symlinked linear-attention conftest initially redirected submodules
back to the original checkout. The first attempted GREEN run (34 failures) is
therefore invalid as a test of the fixed source. It is retained, not hidden.
After copying that source-pinning conftest into the private export, a collection
hook asserts both the served executor path and the rolled main-kernel hash.
The final run passes these assertions before executing all 101 tests.
The earlier report's 89-pass coverage statement is qualified for the same trap;
it is superseded by this source-pinned run. Standalone timing, FP64 paired
audits and NCU do not load the pytest conftest. The earlier backward-only
sanitizer harness explicitly installs its selected modules before invoking
pytest, rather than relying on the exported test directory to select them.

Fixed executor SHA256:
`ebb01ed38721954f356d4403b6ab8f5ac0a9d8ddc0879f1defe3ff2a4e99129d`.
Unchanged rolled main-kernel SHA256:
`37c2a1d53c800954d94295cb5a3eb6afdf9ba2341ab4eff9e1d5ed6d84f181d8`.

Raw logs are retained in the test host's `tune-bh256-20261017/review-comments/`
and the corresponding local `tune-bh256-586/review-comments/` directory:

| Artifact | SHA256 |
| --- | --- |
| `red.log` | `67a248b3473daa032b7fefa0c86364cedb7c6a27b6c560c886a608ff54c1fb95` |
| `green.log` (invalid import scope) | `97bebac8bf8fdd22774df2e3e53e0ad086dcdf14c66a3f45660b65b275fb7681` |
| `green-pinned.log` | `f0bcf306191aac4c82efb6d1d8c7194453cdfee88b8f6f5aca4ebd6efdfe427a` |

## Historical sanitizer log identities

The original local archive contains separate tool and pytest logs. The two
clean memcheck tool logs are each 62 bytes with identical contents, so the
previously equal tool-log hashes are correct. Their pytest logs differ and
contain the different durations. The historical evidence now names both files
and records both hashes; all four were recomputed from the archived bytes and
the pytest summaries checked against their respective logs. No replacement
hash was invented or borrowed from another machine/run.

## Documentation and compatibility

Added missing docstrings in the already-changed benchmark/test helpers. A local
AST scan finds docstrings on all 65 changed Python functions; this is not a
claim about CodeRabbit's own coverage calculation. The additional benchmark,
election-test and resource-test edits are docstring-only by AST comparison.
The PR description follows the submission template; the author explicitly
approved its existing-license checkbox.

Compatibility audit: passed for cold/warm and absent/explicit streams, safe-gate
and grouped-head postpasses, optional inputs/checkpoints and existing execution
regressions. Docstring audit: passed for changed runtime contracts and helpers.
No GPU arithmetic, reduction tree, synchronization or compile-key change is
introduced here; no new performance or sanitizer result is claimed. The prior
full-E2E racecheck limitation remains unchanged. No process was forcibly stopped.
