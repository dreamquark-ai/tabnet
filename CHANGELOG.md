
# [1.2.0](https://github.com/dreamquark-ai/tabnet/compare/v1.1.0...v1.2.0) (2020-07-01)


### Bug Fixes

* verbosity with schedulers ([d6fbf90](https://github.com/dreamquark-ai/tabnet/commit/d6fbf9012ad2a60f0ac4e2b801d258a16250d74c))


### Features

* add entmax as parameter ([96c8a74](https://github.com/dreamquark-ai/tabnet/commit/96c8a74d44abfc7318f06fdd56f7d77ec1974e96))
* allow other optimizer parameters ([16d92d5](https://github.com/dreamquark-ai/tabnet/commit/16d92d5513de36892f859935d36c3bac398ed035))
* allow weights sample for regression ([d40b02f](https://github.com/dreamquark-ai/tabnet/commit/d40b02f5e1cb8ca8c28c398cb0e26cba5cec3445))
* save and load tabnet models ([9d2d8ae](https://github.com/dreamquark-ai/tabnet/commit/9d2d8ae8c724901eb062e4340dad06c364c88fa5))
* save params and easy loading ([6e22393](https://github.com/dreamquark-ai/tabnet/commit/6e22393b9d1206ba1aca8af2645c5de6fe24a6db))



# [1.1.0](https://github.com/dreamquark-ai/tabnet/compare/v1.0.6...v1.1.0) (2020-06-02)


### Bug Fixes

* allow zero layer ([e3b5a04](https://github.com/dreamquark-ai/tabnet/commit/e3b5a04edb1aff25683ce5457f9b4fd57b4c1bf6))
* sort by cat_idx into embedding generator ([9ab3ad5](https://github.com/dreamquark-ai/tabnet/commit/9ab3ad542941ad3ff535f974ad93dc2b950d4559))
* update forest_example notebook ([8092324](https://github.com/dreamquark-ai/tabnet/commit/809232452d5d860036b8e867dfa17701bb2e1c88))


### Features

* add multi output regression ([ffd7c28](https://github.com/dreamquark-ai/tabnet/commit/ffd7c284682f03c1b5a6ce25f910f2d65b78029f))
* add num_workers and drop_last to fit parameters ([313d074](https://github.com/dreamquark-ai/tabnet/commit/313d07481361c87c39df470ee23850757c8b1c85))
* remove mask computations from forward ([44d1a47](https://github.com/dreamquark-ai/tabnet/commit/44d1a47f34c0b9d636279ef3897a02e489471738))
* speed boost and code simplification for GBN ([1642909](https://github.com/dreamquark-ai/tabnet/commit/1642909bd305d40f828be6e0c0484c8f72fd213a))



## [1.0.6](https://github.com/dreamquark-ai/tabnet/compare/v1.0.5...v1.0.6) (2020-04-20)



## [1.0.5](https://github.com/dreamquark-ai/tabnet/compare/v1.0.4...v1.0.5) (2020-03-13)


### Bug Fixes

* remove dead code for plots ([f96795f](https://github.com/dreamquark-ai/tabnet/commit/f96795ff46e02af4ca7c0ed6648276f4e4b788b0))


### Features

* switch to sparse matrix trick ([98910bc](https://github.com/dreamquark-ai/tabnet/commit/98910bcd0424e87208e6520c08726224d214aa33))



## [1.0.4](https://github.com/dreamquark-ai/tabnet/compare/v1.0.3...v1.0.4) (2020-02-28)


### Bug Fixes

* allow smaller different nshared and nindependent ([4b365a7](https://github.com/dreamquark-ai/tabnet/commit/4b365a739d5786c562854eff70042ecc6964bf1a))
* sparsemax on train and predict epoch ([6f7c0e0](https://github.com/dreamquark-ai/tabnet/commit/6f7c0e0211d933d84eeff3e735acad31f0fd70d1))



## [1.0.3](https://github.com/dreamquark-ai/tabnet/compare/v1.0.2...v1.0.3) (2020-02-07)


### Bug Fixes

* map class predictions for XGB results ([3747e2f](https://github.com/dreamquark-ai/tabnet/commit/3747e2f8362174fbf49b7e4890e04427cc4d5fdd))


### Features

* fix shared layers with independent batchnorm ([5f0e43f](https://github.com/dreamquark-ai/tabnet/commit/5f0e43fb961431437d33abe5d70251cf8067d14d))



## [1.0.2](https://github.com/dreamquark-ai/tabnet/compare/v1.0.1...v1.0.2) (2020-02-03)


### Bug Fixes

* multiclass prediction mapper ([2317c5c](https://github.com/dreamquark-ai/tabnet/commit/2317c5cc03c9c9af9e627503fb35934ea6194ce6))
* remove deepcopy from shared blocks ([123932a](https://github.com/dreamquark-ai/tabnet/commit/123932ade14a61a466074269ce7bcf0e61a24613))



## [1.0.1](https://github.com/dreamquark-ai/tabnet/compare/v1.0.0...v1.0.1) (2020-01-20)


### Bug Fixes

* **regression:** fix scheduler ([01e46b7](https://github.com/dreamquark-ai/tabnet/commit/01e46b7b53aa5cb880cca5d1492ef67788c0075e))
* fixing Dockerfile for poetry 1.0 ([6c5cdec](https://github.com/dreamquark-ai/tabnet/commit/6c5cdeca8f3c5a58e2f557f2d8bb5127d3d7f691))
* importance indexing fixed ([a8382c3](https://github.com/dreamquark-ai/tabnet/commit/a8382c31099d59e03c432479b2798abc90f55a58))
* local explain all batches ([91461fb](https://github.com/dreamquark-ai/tabnet/commit/91461fbcd4b8c806e920936e0154258b2dc02373))
* regression gpu integration an typos ([269b4c5](https://github.com/dreamquark-ai/tabnet/commit/269b4c59fcb12d1c24fea7b9e15c7b63aa9939e0))
* resolve timer issue and warnings ([ecd2cd9](https://github.com/dreamquark-ai/tabnet/commit/ecd2cd9c39c1f977868888d6b3abd719a7ee21f4))


### Features

* improve verbosity ([8a2cd87](https://github.com/dreamquark-ai/tabnet/commit/8a2cd8783b4d538648f435798a937a05262a76df))



# [1.0.0](https://github.com/dreamquark-ai/tabnet/compare/v0.1.2...v1.0.0) (2019-12-03)


### Bug Fixes

* **deps:** update dependency numpy to v1.17.3 ([eff6555](https://github.com/dreamquark-ai/tabnet/commit/eff6555ee0b9adbfe90e851eb696cc69df8b2f7d))
* **deps:** update dependency numpy to v1.17.4 ([a80cf29](https://github.com/dreamquark-ai/tabnet/commit/a80cf29cfdb3238518ed73a34b84cd2673272431))
* **deps:** update dependency torch to v1.3.1 ([18ec79b](https://github.com/dreamquark-ai/tabnet/commit/18ec79b879c99671cf756e02a811fee81a915649))
* **deps:** update dependency tqdm to v4.37.0 ([f8f04e7](https://github.com/dreamquark-ai/tabnet/commit/f8f04e783704a204d067c5b67a595e7efc9d7801))
* **deps:** update dependency tqdm to v4.38.0 ([0bf45d2](https://github.com/dreamquark-ai/tabnet/commit/0bf45d26fc241fcfc15e03992a3383f32017ff88))
* functional balanced version ([fab7f16](https://github.com/dreamquark-ai/tabnet/commit/fab7f166a03060a492bc16f78d82ece7f26516b3))
* remove torch warnings (index should be bool) ([f5817cf](https://github.com/dreamquark-ai/tabnet/commit/f5817cfe65d35a4ccb2cba8a147d8696418f09da))


### Features

* add gpu dockerfile and adapt makefile ([8d14406](https://github.com/dreamquark-ai/tabnet/commit/8d14406b9f6b651d6a1fa809c5c2b06ff017422e))
* update notebooks for new model format ([43e2693](https://github.com/dreamquark-ai/tabnet/commit/43e269301c4379ed0daf8f9007ab5048abcbb553))



## [0.1.2](https://github.com/dreamquark-ai/tabnet/compare/e7dc059d8d45ce207b3c24e975dda68fec2155ba...v0.1.2) (2019-11-06)


### Bug Fixes

* add softmax to predict_proba ([bea966f](https://github.com/dreamquark-ai/tabnet/commit/bea966f48ed4521766197f4d424c153f68704733))
* correct code linting ([ae3098c](https://github.com/dreamquark-ai/tabnet/commit/ae3098c0eda62d03f94e52d24a915878f6187100))
* float type when output_dim=1 ([7bb7dfd](https://github.com/dreamquark-ai/tabnet/commit/7bb7dfddb81047503cf44a8d0ae16e14594a7b24))


### Features

* add editorconfig file ([5e84b66](https://github.com/dreamquark-ai/tabnet/commit/5e84b6603ef5c8c5f6fc40b0563c2e9632bb07a2))
* add flake8 ([b1be1f9](https://github.com/dreamquark-ai/tabnet/commit/b1be1f9aa3e822c05094d0483d6269a184360b07))
* run flake8 in CI ([e72d416](https://github.com/dreamquark-ai/tabnet/commit/e72d4160ee46c80dc853c4b3b81bb87ea1bce11d))
* start PyTorch TabNet Paper Implementation ([e7dc059](https://github.com/dreamquark-ai/tabnet/commit/e7dc059d8d45ce207b3c24e975dda68fec2155ba))
