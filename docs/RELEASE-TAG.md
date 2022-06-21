Create a New Tag & Release
===
For every new tag, the workflow at `.github/workflows/release.yml` creates a new release, with the source code and binaries for MacOS, Linux, and Windows.

If you wish to create a new release, do the following:

First:
```
Increase the version number in the Cargo.toml file
```
then:
```
git tag v0.3.1 && git push origin --tag
```
