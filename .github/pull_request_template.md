## Changes

Describe the change and validation.

## Release

To publish binaries, increase the package version in both `Cargo.toml` and
`Cargo.lock`. Merging that increase into `main` tests, builds and publishes
binaries to GitHub Releases.

Changes without a version increase do not publish a release.
