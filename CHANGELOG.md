
# [4.1.0](https://github.com/dreamquark-ai/tabnet/compare/v3.1.1...v4.1.0) (2023-07-23)


### Bug Fixes

* 424 allow any np.intX as training target ([63a8dba](https://github.com/dreamquark-ai/tabnet/commit/63a8dba99e4853b9be5d3e6c14909a30685c7532))
* compute unsupervised loss using numpy ([49bd61b](https://github.com/dreamquark-ai/tabnet/commit/49bd61be4e8faa98ef3b46b4f0115379407e8475))
* custom loss using inplace operations ([423f7c4](https://github.com/dreamquark-ai/tabnet/commit/423f7c43647f8be53f28c9c6061031b7a2644d20))
* disable ansi ([60ec6bf](https://github.com/dreamquark-ai/tabnet/commit/60ec6bf7b27795da44e608d6848573bd0fd4ecd5))
* feature importance not dependent from dataloader ([5b19091](https://github.com/dreamquark-ai/tabnet/commit/5b190916515793114ffa1a9ac4f3869222a14c11))
* README patience to 10 ([fd2c73a](https://github.com/dreamquark-ai/tabnet/commit/fd2c73a4300a745f540a2a789716ec4cabe90a7c))
* replace std 0 by the mean or 1 if mean is 0 ([ddf02da](https://github.com/dreamquark-ai/tabnet/commit/ddf02dab9bdc41c6d7736f0be509950e907909a4))
* try to disable parallel install ([c4963ad](https://github.com/dreamquark-ai/tabnet/commit/c4963ad61e479997c912db816736d073106bcc20))
* typo in pandas error ([5ac5583](https://github.com/dreamquark-ai/tabnet/commit/5ac55834b32693abc4b22028a74475ee0440c2a5))
* update gpg key in docker file gpu ([709fcb1](https://github.com/dreamquark-ai/tabnet/commit/709fcb1ab31f8ac232594877a0d2b3922a02360b))
* upgrade the ressource size ([fc59ea6](https://github.com/dreamquark-ai/tabnet/commit/fc59ea61139228440d2063ead9db42f656d84ff7))
* use numpy std with bessel correction and test ([3adaf4c](https://github.com/dreamquark-ai/tabnet/commit/3adaf4c0858f5d9af8f0f2a2fdaa92360d12cb87))


### Features

* add augmentations inside the fit method ([6d0485f](https://github.com/dreamquark-ai/tabnet/commit/6d0485f58bd1028cffd195d9e27eb97915b9cb2c))
* add warm_start matching scikit-learn ([d725101](https://github.com/dreamquark-ai/tabnet/commit/d725101a559c6be49a6f8e20c3e68b18b8eb7b01))
* added conda install option ([ca14b76](https://github.com/dreamquark-ai/tabnet/commit/ca14b76fc771459745c49723733ff88ef1126d30)), closes [#346](https://github.com/dreamquark-ai/tabnet/issues/346)
* disable tests in docker file gpu to save CI time ([233f74e](https://github.com/dreamquark-ai/tabnet/commit/233f74e41648dad62899ceba7481d58ecfbd87b7))
* enable feature grouping for attention mechanism ([bcae5f4](https://github.com/dreamquark-ai/tabnet/commit/bcae5f43b89fb2c53a0fe8be7c218a7b91afac96))
* enable torch 2.0 by relaxing poetry ([bbd7a4e](https://github.com/dreamquark-ai/tabnet/commit/bbd7a4e96d5503ad23048ce39997462ed1a2eca0))
* pretraining matches paper ([5adb804](https://github.com/dreamquark-ai/tabnet/commit/5adb80482c8242dde7b7942529db94fa9ccbfe48))
* raise error in case cat_dims and cat_idxs are incoherent ([8c3b795](https://github.com/dreamquark-ai/tabnet/commit/8c3b7951642f62e7449bb95875b5265d4b89148e))
* update python ([dea62b4](https://github.com/dreamquark-ai/tabnet/commit/dea62b410e3f4cc729f1c1933018d7d8db24d016))



## [3.1.1](https://github.com/dreamquark-ai/tabnet/compare/v3.1.0...v3.1.1) (2021-02-02)


### Bug Fixes

* add preds_mapper to pretraining ([76f2c85](https://github.com/dreamquark-ai/tabnet/commit/76f2c852f59c6ed2c5dc5f0766cb99310bae5f2c))



# [3.1.0](https://github.com/dreamquark-ai/tabnet/compare/v3.0.0...v3.1.0) (2021-01-12)


### Bug Fixes

* n_a not being used ([7ae20c9](https://github.com/dreamquark-ai/tabnet/commit/7ae20c98a601da95040b9ecf79eac19f1d3e4a7b))


### Features

* save and load preds_mapper ([cab643b](https://github.com/dreamquark-ai/tabnet/commit/cab643b156fdecfded51d70d29072fc43f397bbb))



# [3.0.0](https://github.com/dreamquark-ai/tabnet/compare/v2.0.1...v3.0.0) (2020-12-15)


### Bug Fixes

* checknan allow string as targets ([855befc](https://github.com/dreamquark-ai/tabnet/commit/855befc5a2cd153509b8c93eccdea866bf094a29))
* deactivate pin memory when device is cpu ([bd0b96f](https://github.com/dreamquark-ai/tabnet/commit/bd0b96f4f61c44b58713f60a030094cc21edb6e3))
* fixed docstring issues ([d216fbf](https://github.com/dreamquark-ai/tabnet/commit/d216fbfa4dadd6c8d4110fa8da0f1c0bdfdccc2d))
* load from cpu when saved on gpu ([451bd86](https://github.com/dreamquark-ai/tabnet/commit/451bd8669038ddf7869843f45ca872ae92e2260d))


### Features

* add new default metrics ([0fe5b72](https://github.com/dreamquark-ai/tabnet/commit/0fe5b72b60e894fae821488c0d4c34752309fc26))
* enable self supervised pretraining ([d4af838](https://github.com/dreamquark-ai/tabnet/commit/d4af838d375128b3d62e17622ec8e0a558faf1b7))
* mask-dependent loss ([64052b0](https://github.com/dreamquark-ai/tabnet/commit/64052b0f816eb9d63008347783cd1fe655be3088))



## [2.0.1](https://github.com/dreamquark-ai/tabnet/compare/v2.0.0...v2.0.1) (2020-10-15)


### Bug Fixes

* add check for evalset dim ([ba09980](https://github.com/dreamquark-ai/tabnet/commit/ba09980029093ddfee3f10414c366893ea0e4005))
* pin memory available for training only ([28346c2](https://github.com/dreamquark-ai/tabnet/commit/28346c2259cabbed79e83963c4009eac3ae38f9e))
* specify device ([46a301f](https://github.com/dreamquark-ai/tabnet/commit/46a301fc5ae702f56f2f54ccabf61762da26588d))
* torch.load map_location in Py36 fallback ([63cb8c4](https://github.com/dreamquark-ai/tabnet/commit/63cb8c43652f854b0e11a6c8f784d4b5f8f8748b))



# [2.0.0](https://github.com/dreamquark-ai/tabnet/compare/v1.2.0...v2.0.0) (2020-10-13)


### Bug Fixes

* 1000 lines only when env=CI ([c557349](https://github.com/dreamquark-ai/tabnet/commit/c5573496e1262bc765eb04361ae4a3185844a866))
* add map_location to torch load ([c2b560e](https://github.com/dreamquark-ai/tabnet/commit/c2b560e72bc01e34e8dba7578f239e37bbd6782c))
* load_model fallback to BytesIO for Py3.6 ([55c09e5](https://github.com/dreamquark-ai/tabnet/commit/55c09e5c47e6ec58276c301a5af7afa2dc529bc1))


### Features

* add check nan and inf ([d871406](https://github.com/dreamquark-ai/tabnet/commit/d87140623f2118e494874549752987e89be235f3))
* add easy schedulers ([0ae114f](https://github.com/dreamquark-ai/tabnet/commit/0ae114ff59900537cd3c48dc9d44669f52b9141e))
* adding callbacks and metrics ([1e0daec](https://github.com/dreamquark-ai/tabnet/commit/1e0daec01a6a95f39699028c5fad213b2d8f3d3e))
* refacto models with metrics and callbacks ([cc57d62](https://github.com/dreamquark-ai/tabnet/commit/cc57d62698ef629d63dcc8878d4d48f231f3cd77))
* speedups ([5a01359](https://github.com/dreamquark-ai/tabnet/commit/5a013596da597263aaf1b9f385732fc2442dda96))
* TabNetMultiTaskClassifier ([5764a43](https://github.com/dreamquark-ai/tabnet/commit/5764a43e72cb643fff806f70ed9dfa2e48433f50))
* update readme and notebooks ([9cb38d2](https://github.com/dreamquark-ai/tabnet/commit/9cb38d2d3b636ef5f0a99a9ac4171faeea141213))



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

* fixing Dockerfile for poetry 1.0 ([6c5cdec](https://github.com/dreamquark-ai/tabnet/commit/6c5cdeca8f3c5a58e2f557f2d8bb5127d3d7f691))
* importance indexing fixed ([a8382c3](https://github.com/dreamquark-ai/tabnet/commit/a8382c31099d59e03c432479b2798abc90f55a58))
* local explain all batches ([91461fb](https://github.com/dreamquark-ai/tabnet/commit/91461fbcd4b8c806e920936e0154258b2dc02373))
* regression gpu integration an typos ([269b4c5](https://github.com/dreamquark-ai/tabnet/commit/269b4c59fcb12d1c24fea7b9e15c7b63aa9939e0))
* **regression:** fix scheduler ([01e46b7](https://github.com/dreamquark-ai/tabnet/commit/01e46b7b53aa5cb880cca5d1492ef67788c0075e))
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
