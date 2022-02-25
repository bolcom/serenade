Create a New Tag & Release
===
For every new tag, the workflow at `.github/workflows/release.yml` creates a new release, with the source code and binaries for MacOS and Linux.

If you wish to create a new release, just push a new git tag in the following format:
```
git tag v<MAJOR>.<MINOR>.<PATCH> && git push origin --tag
```

For example:
```
git tag v0.0.2 && git push origin --tag
```
