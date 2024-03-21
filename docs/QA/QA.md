# QA

背景：我正在使用hyperImpute开源库，进行数据插补实验。我使用教程，使用自己的数据替换进行训练。现在正在进行该存储库中的已有的插补方法的测试，测试这些方法对我的数据集的插补性能，去选择一个最优的插补方法。

问题：在使用自己的数据集训练过程中，发现训练的速度很慢，特别是 fit_transform() 方法。我想知道如何加速训练过程。更好的利用本机器的CPU，GPU，内存等资源，提高训练速度。

源代码的部分内容：

    ```python
    pct = 0.3

    mechanisms = ["MAR", "MNAR", "MCAR"]
    percentages = [pct]

    plugins = ['most_frequent',
    'sinkhorn',
    'softimpute',
    'EM',
    'sklearn_missforest',
    'miracle',
    'nop',
    'hyperimpute',
    'mice',
    'sklearn_ice',
    'median',
    'ice',
    'missforest',
    'miwae',
    'mean',
    'gain']

    for plugin in tqdm(plugins):

        for ampute_mechanism in mechanisms:
            for p_miss in percentages:                
                ctx = imputers.get(plugin)
                x, x_miss, mask = datasets[ampute_mechanism][p_miss]
                start = time.time() * 1000
                x_imp = ctx.fit_transform(x_miss)
                end = time.time() * 1000
                print(f"Time cost: {end - start} ms")
    ```

这是我搜索出来的一些关于fit_transform()方法的源代码文件内容：

    ```text
    36 个结果 - 19 文件

    src\hyperimpute\plugins\core\base_plugin.py:
    113  
    114:     def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    115          return pd.DataFrame(self.fit(X, *args, *kwargs).transform(X))

    src\hyperimpute\plugins\imputers\_hyperimpute_internals.py:
    702                  le = LabelEncoder()
    703:                 X.loc[X[col].notnull(), col] = le.fit_transform(existing_vals).astype(
    704                      int

    856          # Use baseline imputer for initial values
    857:         return self.baseline_imputer.fit_transform(X)
    858  

    867          return pd.DataFrame(
    868:             MissingIndicator(features="all").fit_transform(X),
    869              columns=X.columns,

    873      @validate_arguments(config=dict(arbitrary_types_allowed=True))
    874:     def _fit_transform_inner_optimization(self, X: pd.DataFrame) -> pd.DataFrame:
    875          log.info("  > HyperImpute using inner optimization")

    915      @validate_arguments(config=dict(arbitrary_types_allowed=True))
    916:     def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
    917          # Run imputation

    924  
    925:         Xt = self._fit_transform_inner_optimization(Xt)
    926  

    src\hyperimpute\plugins\imputers\plugin_EM.py:
    42      @decorators.expect_ndarray_for(1)
    43:     def fit_transform(self, X: np.ndarray) -> np.ndarray:
    44          """Imputes the provided dataset using the EM strategy.

    192              # fallback to mean imputation in case of singular matrix.
    193:             X_reconstructed = SimpleImputer(strategy="mean").fit_transform(
    194                  X_reconstructed

    222          >>> plugin = Imputers().get("EM")
    223:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    224  

    243      def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
    244:         return self._model.fit_transform(X.to_numpy())
    245  

    src\hyperimpute\plugins\imputers\plugin_gain.py:
    305  
    306:     def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
    307          """Imputes the provided dataset using the GAIN strategy.

    328          >>> plugin = Imputers().get("gain")
    329:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    330      """

    src\hyperimpute\plugins\imputers\plugin_hyperimpute.py:
    49          >>> plugin = Imputers().get("hyperimpute")
    50:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    51  

    127      def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
    128:         return self.model.fit_transform(X)
    129  

    src\hyperimpute\plugins\imputers\plugin_ice.py:
    28          >>> plugin = Imputers().get("ice")
    29:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    30  

    src\hyperimpute\plugins\imputers\plugin_mean.py:
    22          >>> plugin = Imputers().get("mean")
    23:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    24      """

    src\hyperimpute\plugins\imputers\plugin_median.py:
    22          >>> plugin = Imputers().get("median")
    23:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    24               0    1    2    3

    src\hyperimpute\plugins\imputers\plugin_mice.py:
    36          >>> plugin = Imputers().get("mice")
    37:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    38      """

    src\hyperimpute\plugins\imputers\plugin_miracle.py:
    24          >>> plugin = Imputers().get("miracle")
    25:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    26  

    116          seed_imputer = self._get_seed_imputer(self.seed_imputation)
    117:         X_seed = seed_imputer.fit_transform(X)
    118  

    src\hyperimpute\plugins\imputers\plugin_missforest.py:
    37          >>> plugin = Imputers().get("missforest")
    38:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    39      """

    src\hyperimpute\plugins\imputers\plugin_miwae.py:
    46          >>> plugin = Imputers().get("miwae")
    47:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    48  

    src\hyperimpute\plugins\imputers\plugin_most_frequent.py:
    22          >>> plugin = Imputers().get("most_frequent")
    23:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    24      """

    src\hyperimpute\plugins\imputers\plugin_sinkhorn.py:
    43          >>> plugin = Imputers().get("sinkhorn")
    44:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    45  

    72  
    73:     def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    74          X = torch.tensor(X.values).to(DEVICE)

    129          >>> plugin = Imputers().get("sinkhorn")
    130:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    131                    0         1         2         3

    194      def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
    195:         return self._model.fit_transform(X)
    196  

    src\hyperimpute\plugins\imputers\plugin_sklearn_ice.py:
    29          >>> plugin = Imputers().get("ice")
    30:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    31      """

    src\hyperimpute\plugins\imputers\plugin_sklearn_missforest.py:
    36          >>> plugin = Imputers().get("sklearn_missforest")
    37:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    38               0    1    2    3

    src\hyperimpute\plugins\imputers\plugin_softimpute.py:
    40          >>> plugin = Imputers().get("softimpute")
    41:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    42  

    115      @decorators.expect_ndarray_for(1)
    116:     def fit_transform(self, X: np.ndarray, **fit_params: Any) -> np.ndarray:
    117          return self.fit(X, **fit_params).transform(X)

    272          >>> plugin = Imputers().get("softimpute")
    273:         >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    274                        0             1             2             3

    src\hyperimpute\plugins\prediction\classifiers\plugin_xgboost.py:
    136          self.encoder = LabelEncoder()
    137:         y = self.encoder.fit_transform(y)
    138          self.model.fit(X, y, **kwargs)

    src\hyperimpute\utils\benchmarks.py:
    73      cols = X.columns
    74:     return pd.DataFrame(preproc.fit_transform(X), columns=cols)
    75  

    122  
    123:     imputed = model.fit_transform(X_miss.copy())
    124  

    ```

要求：

1. 研究生水平；
2. 清楚简洁，抓住核心，专注于问题解决方案
