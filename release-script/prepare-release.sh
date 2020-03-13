#!/bin/bash

# stop if error
set -e

read -p 'Release version: ' version
echo ${version} | grep v && echo "Version should be x.y.z (for example, 1.1.1, 2.0.0, ...)" && exit -1

localDir=`readlink -f .`
releaseDir="${localDir}/release-${version}"
rm -rf ${releaseDir}
mkdir ${releaseDir}
cd $releaseDir

echo "Cloning repo into tabnet"
git clone -q git@github.com:dreamquark-ai/tabnet.git tabnet

cd tabnet
# Create release branch and push it
git checkout -b release/${version}

# Change version of package
docker run --rm -v ${PWD}:/work -w /work tabnet:latest poetry version ${version}
# Add modified file
git add pyproject.toml
# Commit release
git commit -m "chore: release v${version}"
# Create tag for changelog generation
git tag v${version}
docker run -v ${PWD}:/work -w /work --entrypoint "" release-changelog:latest conventional-changelog -p angular -i CHANGELOG.md -s -r 0
docker run -v ${PWD}:/work -w /work --entrypoint "" release-changelog:latest chmod 777 CHANGELOG.md
# Removing 4 first line of the file
echo "$(tail -n +4 CHANGELOG.md)" > CHANGELOG.md
# Deleting tag 
git tag -d v${version}
# Adding CHANGELOG to commit
git add CHANGELOG.md
git commit --amend --no-edit
# Push release branch
git push origin release/${version}

cd ${localDir}
rm -rf ${releaseDir}