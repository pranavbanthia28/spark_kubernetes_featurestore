"""
Spark DataFrame Operations Module

This module provides a comprehensive set of utilities for working with PySpark DataFrames,
including transformation, optimization, and analysis operations.

Example Usage:
    ```python
    from spark_dataframe_ops import DataFrameTransformer
    from pyspark.sql import SparkSession
    
    # Create a SparkSession
    spark = SparkSession.builder.appName("DataFrameOpsExample").getOrCreate()
    
    # Create a sample DataFrame
    data = [
        (1, "John", 30, "Sales"),
        (2, "Jane", 25, "Engineering"),
        (3, "Bob", 40, "Sales"),
        (4, "Alice", 35, "Engineering")
    ]
    df = spark.createDataFrame(data, ["id", "name", "age", "department"])
    
    # Create a transformer
    transformer = DataFrameTransformer(df)
    
    # Apply transformations
    filtered_df = transformer.filter_by_condition("age > 30")
    aggregated_df = transformer.aggregate_by(["department"], {"age": "avg"})
    
    # Show results
    filtered_df.show()
    aggregated_df.show()
    ```
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable

from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFrameOperationError(Exception):
    """Exception raised for errors in DataFrame operations."""
    pass


class DataFrameTransformer:
    """
    A class for applying common transformations to PySpark DataFrames.
    
    This class provides methods for filtering, joining, aggregating, and applying
    window functions to DataFrames, with proper error handling and logging.
    """
    
    def __init__(self, dataframe: DataFrame):
        """
        Initialize the transformer with a DataFrame.
        
        Args:
            dataframe: The DataFrame to transform
        """
        self.df = dataframe
        self.spark = dataframe.sparkSession
    
    def filter_by_condition(self, condition: str) -> DataFrame:
        """
        Filter the DataFrame using a SQL-like condition string.
        
        Args:
            condition: SQL-like condition string (e.g., "age > 30")
            
        Returns:
            DataFrame: Filtered DataFrame
            
        Raises:
            DataFrameOperationError: If the filtering operation fails
        """
        try:
            result_df = self.df.filter(condition)
            logger.info(f"Applied filter condition: {condition}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to apply filter condition: {str(e)}")
            raise DataFrameOperationError(f"Filter operation failed: {str(e)}")
    
    def filter_by_column_values(
        self,
        column: str,
        values: List[Any],
        include: bool = True
    ) -> DataFrame:
        """
        Filter the DataFrame to include or exclude rows with specific column values.
        
        Args:
            column: Column name to filter on
            values: List of values to include or exclude
            include: If True, include rows with values in the list; if False, exclude them
            
        Returns:
            DataFrame: Filtered DataFrame
            
        Raises:
            DataFrameOperationError: If the filtering operation fails
        """
        try:
            if include:
                result_df = self.df.filter(F.col(column).isin(values))
                logger.info(f"Filtered to include rows where {column} is in {values}")
            else:
                result_df = self.df.filter(~F.col(column).isin(values))
                logger.info(f"Filtered to exclude rows where {column} is in {values}")
            
            return result_df
        except Exception as e:
            logger.error(f"Failed to filter by column values: {str(e)}")
            raise DataFrameOperationError(f"Filter by column values failed: {str(e)}")
    
    def join_with(
        self,
        other_df: DataFrame,
        join_cols: Union[str, List[str]],
        join_type: str = "inner"
    ) -> DataFrame:
        """
        Join the DataFrame with another DataFrame.
        
        Args:
            other_df: DataFrame to join with
            join_cols: Column(s) to join on
            join_type: Type of join (inner, outer, left, right, left_semi, left_anti)
            
        Returns:
            DataFrame: Joined DataFrame
            
        Raises:
            DataFrameOperationError: If the join operation fails
        """
        try:
            result_df = self.df.join(other_df, on=join_cols, how=join_type)
            logger.info(f"Joined DataFrames on {join_cols} using {join_type} join")
            return result_df
        except Exception as e:
            logger.error(f"Failed to join DataFrames: {str(e)}")
            raise DataFrameOperationError(f"Join operation failed: {str(e)}")
    
    def aggregate_by(
        self,
        group_cols: List[str],
        agg_exprs: Dict[str, str]
    ) -> DataFrame:
        """
        Perform groupBy aggregation on the DataFrame.
        
        Args:
            group_cols: Columns to group by
            agg_exprs: Dictionary mapping column names to aggregation functions
                      (e.g., {"age": "avg", "salary": "sum"})
            
        Returns:
            DataFrame: Aggregated DataFrame
            
        Raises:
            DataFrameOperationError: If the aggregation operation fails
        """
        try:
            # Create the groupBy object
            grouped = self.df.groupBy(*group_cols)
            
            # Build the list of aggregation expressions
            agg_cols = []
            for col, agg_func in agg_exprs.items():
                agg_cols.append(F.expr(f"{agg_func}({col})").alias(f"{col}_{agg_func}"))
            
            # Apply the aggregation
            result_df = grouped.agg(*agg_cols)
            
            logger.info(f"Performed groupBy on {group_cols} with aggregations: {agg_exprs}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to perform aggregation: {str(e)}")
            raise DataFrameOperationError(f"Aggregation operation failed: {str(e)}")
    
    def apply_window_function(
        self,
        partition_cols: List[str],
        order_cols: List[str],
        window_exprs: Dict[str, str]
    ) -> DataFrame:
        """
        Apply window functions to the DataFrame.
        
        Args:
            partition_cols: Columns to partition by
            order_cols: Columns to order by
            window_exprs: Dictionary mapping output column names to window function expressions
                         (e.g., {"rank": "rank()", "running_sum": "sum(amount) over (order by date)"})
            
        Returns:
            DataFrame: DataFrame with window functions applied
            
        Raises:
            DataFrameOperationError: If the window function operation fails
        """
        try:
            # Create the window specification
            window_spec = Window.partitionBy(*partition_cols).orderBy(*order_cols)
            
            # Apply the window functions
            result_df = self.df
            for output_col, window_expr in window_exprs.items():
                result_df = result_df.withColumn(
                    output_col,
                    F.expr(window_expr).over(window_spec)
                )
            
            logger.info(f"Applied window functions: {window_exprs}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to apply window functions: {str(e)}")
            raise DataFrameOperationError(f"Window function operation failed: {str(e)}")
    
    def select_columns(self, columns: List[str]) -> DataFrame:
        """
        Select specific columns from the DataFrame.
        
        Args:
            columns: List of column names to select
            
        Returns:
            DataFrame: DataFrame with only the selected columns
            
        Raises:
            DataFrameOperationError: If the select operation fails
        """
        try:
            result_df = self.df.select(*columns)
            logger.info(f"Selected columns: {columns}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to select columns: {str(e)}")
            raise DataFrameOperationError(f"Select operation failed: {str(e)}")
    
    def rename_columns(self, rename_map: Dict[str, str]) -> DataFrame:
        """
        Rename columns in the DataFrame.
        
        Args:
            rename_map: Dictionary mapping old column names to new column names
            
        Returns:
            DataFrame: DataFrame with renamed columns
            
        Raises:
            DataFrameOperationError: If the rename operation fails
        """
        try:
            result_df = self.df
            for old_name, new_name in rename_map.items():
                result_df = result_df.withColumnRenamed(old_name, new_name)
            
            logger.info(f"Renamed columns: {rename_map}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to rename columns: {str(e)}")
            raise DataFrameOperationError(f"Rename operation failed: {str(e)}")
    
    def add_columns(self, column_exprs: Dict[str, str]) -> DataFrame:
        """
        Add new columns to the DataFrame using expressions.
        
        Args:
            column_exprs: Dictionary mapping new column names to expressions
                         (e.g., {"full_name": "concat(first_name, ' ', last_name)"})
            
        Returns:
            DataFrame: DataFrame with added columns
            
        Raises:
            DataFrameOperationError: If the add columns operation fails
        """
        try:
            result_df = self.df
            for col_name, expr in column_exprs.items():
                result_df = result_df.withColumn(col_name, F.expr(expr))
            
            logger.info(f"Added columns: {column_exprs}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to add columns: {str(e)}")
            raise DataFrameOperationError(f"Add columns operation failed: {str(e)}")
    
    def drop_columns(self, columns: List[str]) -> DataFrame:
        """
        Drop columns from the DataFrame.
        
        Args:
            columns: List of column names to drop
            
        Returns:
            DataFrame: DataFrame with dropped columns
            
        Raises:
            DataFrameOperationError: If the drop columns operation fails
        """
        try:
            result_df = self.df.drop(*columns)
            logger.info(f"Dropped columns: {columns}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to drop columns: {str(e)}")
            raise DataFrameOperationError(f"Drop columns operation failed: {str(e)}")
    
    def sort_by(
        self,
        columns: List[str],
        ascending: Union[bool, List[bool]] = True
    ) -> DataFrame:
        """
        Sort the DataFrame by specified columns.
        
        Args:
            columns: List of column names to sort by
            ascending: Whether to sort in ascending order (True) or descending order (False)
                      Can be a single boolean or a list of booleans (one per column)
            
        Returns:
            DataFrame: Sorted DataFrame
            
        Raises:
            DataFrameOperationError: If the sort operation fails
        """
        try:
            # Create sort expressions based on ascending parameter
            if isinstance(ascending, bool):
                # Same direction for all columns
                sort_exprs = [F.col(col).asc() if ascending else F.col(col).desc() for col in columns]
            else:
                # Different direction for each column
                if len(ascending) != len(columns):
                    raise ValueError("Length of ascending list must match length of columns list")
                
                sort_exprs = [
                    F.col(col).asc() if asc else F.col(col).desc()
                    for col, asc in zip(columns, ascending)
                ]
            
            result_df = self.df.sort(*sort_exprs)
            logger.info(f"Sorted by columns: {columns} with ascending: {ascending}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to sort DataFrame: {str(e)}")
            raise DataFrameOperationError(f"Sort operation failed: {str(e)}")
    
    def limit_rows(self, n: int) -> DataFrame:
        """
        Limit the number of rows in the DataFrame.
        
        Args:
            n: Maximum number of rows to include
            
        Returns:
            DataFrame: DataFrame with at most n rows
            
        Raises:
            DataFrameOperationError: If the limit operation fails
        """
        try:
            result_df = self.df.limit(n)
            logger.info(f"Limited to {n} rows")
            return result_df
        except Exception as e:
            logger.error(f"Failed to limit rows: {str(e)}")
            raise DataFrameOperationError(f"Limit operation failed: {str(e)}")
    
    def union_with(self, other_df: DataFrame) -> DataFrame:
        """
        Union the DataFrame with another DataFrame.
        
        Args:
            other_df: DataFrame to union with
            
        Returns:
            DataFrame: Unioned DataFrame
            
        Raises:
            DataFrameOperationError: If the union operation fails
        """
        try:
            result_df = self.df.union(other_df)
            logger.info("Performed union operation")
            return result_df
        except Exception as e:
            logger.error(f"Failed to union DataFrames: {str(e)}")
            raise DataFrameOperationError(f"Union operation failed: {str(e)}")
    
    def intersect_with(self, other_df: DataFrame) -> DataFrame:
        """
        Intersect the DataFrame with another DataFrame.
        
        Args:
            other_df: DataFrame to intersect with
            
        Returns:
            DataFrame: Intersected DataFrame
            
        Raises:
            DataFrameOperationError: If the intersect operation fails
        """
        try:
            result_df = self.df.intersect(other_df)
            logger.info("Performed intersect operation")
            return result_df
        except Exception as e:
            logger.error(f"Failed to intersect DataFrames: {str(e)}")
            raise DataFrameOperationError(f"Intersect operation failed: {str(e)}")
    
    def except_with(self, other_df: DataFrame) -> DataFrame:
        """
        Perform except operation with another DataFrame.
        
        Args:
            other_df: DataFrame to except with
            
        Returns:
            DataFrame: Result of except operation
            
        Raises:
            DataFrameOperationError: If the except operation fails
        """
        try:
            result_df = self.df.exceptAll(other_df)
            logger.info("Performed except operation")
            return result_df
        except Exception as e:
            logger.error(f"Failed to perform except operation: {str(e)}")
            raise DataFrameOperationError(f"Except operation failed: {str(e)}")
    
    def with_column_cast(
        self,
        column_types: Dict[str, str]
    ) -> DataFrame:
        """
        Cast columns to specified types.
        
        Args:
            column_types: Dictionary mapping column names to type names
                         (e.g., {"age": "int", "salary": "double"})
            
        Returns:
            DataFrame: DataFrame with cast columns
            
        Raises:
            DataFrameOperationError: If the cast operation fails
        """
        try:
            result_df = self.df
            for col_name, type_name in column_types.items():
                result_df = result_df.withColumn(col_name, F.col(col_name).cast(type_name))
            
            logger.info(f"Cast columns to types: {column_types}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to cast columns: {str(e)}")
            raise DataFrameOperationError(f"Cast operation failed: {str(e)}")
    
    def cache_with_strategy(
        self,
        storage_level: str = "MEMORY_AND_DISK"
    ) -> DataFrame:
        """
        Cache the DataFrame with a specified storage level.
        
        This method allows for fine-grained control over how the DataFrame is cached,
        using different persistence levels to optimize for memory usage, disk usage,
        serialization, and replication.
        
        Args:
            storage_level: The storage level to use for caching. Options include:
                - "MEMORY_ONLY": Store RDD as deserialized Java objects in the JVM.
                - "MEMORY_AND_DISK": Store RDD as deserialized Java objects in the JVM. 
                                    If the RDD doesn't fit in memory, store the partitions 
                                    that don't fit on disk, and read them from there when needed.
                - "MEMORY_ONLY_SER": Store RDD as serialized Java objects (one byte array per partition).
                - "MEMORY_AND_DISK_SER": Similar to MEMORY_ONLY_SER, but spill partitions that don't 
                                        fit in memory to disk instead of recomputing them on the fly.
                - "DISK_ONLY": Store the RDD partitions only on disk.
                - "OFF_HEAP": Store RDD in serialized format in Tachyon.
            
        Returns:
            DataFrame: The cached DataFrame
            
        Raises:
            DataFrameOperationError: If the caching operation fails
        """
        try:
            from pyspark.storagelevel import StorageLevel
            
            # Map string storage level to StorageLevel object
            storage_level_map = {
                "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
                "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
                "MEMORY_ONLY_SER": StorageLevel.MEMORY_ONLY_SER,
                "MEMORY_AND_DISK_SER": StorageLevel.MEMORY_AND_DISK_SER,
                "DISK_ONLY": StorageLevel.DISK_ONLY,
                "OFF_HEAP": StorageLevel.OFF_HEAP
            }
            
            if storage_level not in storage_level_map:
                raise ValueError(f"Invalid storage level: {storage_level}. "
                                f"Valid options are: {list(storage_level_map.keys())}")
            
            # Persist with the specified storage level
            result_df = self.df.persist(storage_level_map[storage_level])
            
            logger.info(f"Cached DataFrame with storage level: {storage_level}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to cache DataFrame: {str(e)}")
            raise DataFrameOperationError(f"Cache operation failed: {str(e)}")
    
    def optimize_query_execution(
        self,
        enable_adaptive_execution: bool = True,
        enable_broadcast_join: bool = True,
        broadcast_threshold: Optional[int] = None,
        auto_broadcast_join_threshold: Optional[int] = None
    ) -> DataFrame:
        """
        Apply various Spark SQL optimizations to improve query execution performance.
        
        This method configures Spark SQL optimization parameters for the current DataFrame
        to improve query execution performance. It can enable/disable adaptive query execution,
        broadcast joins, and set thresholds for broadcast operations.
        
        Args:
            enable_adaptive_execution: Whether to enable adaptive query execution
            enable_broadcast_join: Whether to enable broadcast join optimization
            broadcast_threshold: Size in bytes below which a table will be broadcast 
                                when performing a join (for explicit broadcasts)
            auto_broadcast_join_threshold: Size in bytes below which a table will 
                                          be broadcast automatically when performing a join
            
        Returns:
            DataFrame: The optimized DataFrame
            
        Raises:
            DataFrameOperationError: If the optimization operation fails
        """
        try:
            # Get the SparkSession
            spark = self.spark
            
            # Create a new DataFrame with the same data but with optimized execution
            result_df = self.df
            
            # Configure adaptive execution
            spark.conf.set("spark.sql.adaptive.enabled", enable_adaptive_execution)
            
            # Configure broadcast join optimization
            spark.conf.set("spark.sql.join.preferSortMergeJoin", not enable_broadcast_join)
            
            # Set broadcast thresholds if provided
            if broadcast_threshold is not None:
                spark.conf.set("spark.sql.broadcastTimeout", broadcast_threshold)
            
            if auto_broadcast_join_threshold is not None:
                spark.conf.set("spark.sql.autoBroadcastJoinThreshold", auto_broadcast_join_threshold)
            
            # Log the optimization settings
            logger.info(f"Applied query optimizations: adaptive_execution={enable_adaptive_execution}, "
                       f"broadcast_join={enable_broadcast_join}")
            
            return result_df
        except Exception as e:
            logger.error(f"Failed to optimize query execution: {str(e)}")
            raise DataFrameOperationError(f"Query optimization failed: {str(e)}")
    
    def repartition_optimize(
        self,
        num_partitions: Optional[int] = None,
        partition_cols: Optional[List[str]] = None,
        coalesce_only: bool = False,
        target_size_per_partition: Optional[int] = None
    ) -> DataFrame:
        """
        Optimize the DataFrame partitioning for better performance.
        
        This method provides options to repartition or coalesce the DataFrame to optimize
        data distribution across partitions. It can repartition by a specific number of partitions,
        by specific columns, or coalesce to reduce the number of partitions.
        
        Args:
            num_partitions: Target number of partitions (if None and partition_cols is None, 
                           the current partitioning is preserved)
            partition_cols: Columns to partition by (for hash partitioning)
            coalesce_only: If True, use coalesce instead of repartition to avoid a full shuffle
            target_size_per_partition: Target size in bytes for each partition 
                                      (used to calculate num_partitions if not provided)
            
        Returns:
            DataFrame: The repartitioned DataFrame
            
        Raises:
            DataFrameOperationError: If the repartitioning operation fails
        """
        try:
            result_df = self.df
            
            # Calculate number of partitions based on target size if provided
            if num_partitions is None and target_size_per_partition is not None:
                # Get the estimated size of the DataFrame
                estimated_size = result_df.count() * len(result_df.columns) * 8  # Rough estimate
                num_partitions = max(1, estimated_size // target_size_per_partition)
            
            # Apply the appropriate partitioning strategy
            if coalesce_only and num_partitions is not None:
                # Use coalesce to reduce partitions without a full shuffle
                current_partitions = result_df.rdd.getNumPartitions()
                if current_partitions > num_partitions:
                    result_df = result_df.coalesce(num_partitions)
                    logger.info(f"Coalesced DataFrame from {current_partitions} to {num_partitions} partitions")
            elif partition_cols is not None:
                # Repartition by columns (hash partitioning)
                if num_partitions is not None:
                    result_df = result_df.repartition(num_partitions, *partition_cols)
                    logger.info(f"Repartitioned DataFrame to {num_partitions} partitions by columns: {partition_cols}")
                else:
                    result_df = result_df.repartition(*partition_cols)
                    logger.info(f"Repartitioned DataFrame by columns: {partition_cols}")
            elif num_partitions is not None:
                # Repartition by number only
                result_df = result_df.repartition(num_partitions)
                logger.info(f"Repartitioned DataFrame to {num_partitions} partitions")
            
            return result_df
        except Exception as e:
            logger.error(f"Failed to optimize partitioning: {str(e)}")
            raise DataFrameOperationError(f"Partition optimization failed: {str(e)}")
    
    def apply_udf_transformation(
        self,
        udf_manager: UDFManager,
        udf_name: str,
        input_cols: Union[str, List[str]],
        output_col: str,
        use_sql: bool = False,
        sql_expr: Optional[str] = None
    ) -> DataFrame:
        """
        Apply a UDF transformation to the DataFrame using a UDFManager.
        
        This method integrates with the UDFManager to apply UDFs directly from the transformer.
        It supports both regular UDF application and SQL-based UDF application.
        
        Args:
            udf_manager: UDFManager instance containing registered UDFs
            udf_name: Name of the registered UDF to apply
            input_cols: Column(s) to use as input to the UDF (for non-SQL mode)
            output_col: Column name for the UDF output (for non-SQL mode)
            use_sql: Whether to use SQL expression mode
            sql_expr: SQL expression to use (required if use_sql is True)
            
        Returns:
            DataFrame: DataFrame with the UDF applied
            
        Raises:
            DataFrameOperationError: If the UDF application fails
            ValueError: If invalid parameters are provided
        """
        try:
            if use_sql:
                if not sql_expr:
                    raise ValueError("SQL expression must be provided when use_sql is True")
                
                result_df = udf_manager.apply_sql_udf(
                    df=self.df,
                    udf_name=udf_name,
                    sql_expr=sql_expr
                )
            else:
                result_df = udf_manager.apply_udf(
                    df=self.df,
                    udf_name=udf_name,
                    input_cols=input_cols,
                    output_col=output_col
                )
            
            logger.info(f"Applied UDF transformation: {udf_name}")
            return result_df
        except Exception as e:
            logger.error(f"Failed to apply UDF transformation: {str(e)}")
            raise DataFrameOperationError(f"UDF transformation failed: {str(e)}")
    
    def apply_multiple_udfs(
        self,
        udf_manager: UDFManager,
        udf_configs: List[Dict[str, Any]]
    ) -> DataFrame:
        """
        Apply multiple UDFs to the DataFrame in a single pass.
        
        This method integrates with the UDFManager to apply multiple UDFs in sequence.
        
        Args:
            udf_manager: UDFManager instance containing registered UDFs
            udf_configs: List of dictionaries, each containing:
                        - 'udf_name': Name of the registered UDF
                        - 'input_cols': Column(s) to use as input
                        - 'output_col': Column name for the output
            
        Returns:
            DataFrame: DataFrame with all UDFs applied
            
        Raises:
            DataFrameOperationError: If the UDF application fails
        """
        try:
            result_df = udf_manager.batch_apply_udfs(
                df=self.df,
                udf_configs=udf_configs
            )
            
            logger.info(f"Applied {len(udf_configs)} UDF transformations")
            return result_df
        except Exception as e:
            logger.error(f"Failed to apply multiple UDF transformations: {str(e)}")
            raise DataFrameOperationError(f"Multiple UDF transformations failed: {str(e)}")


class UDFManager:
    """
    A manager for registering and applying User Defined Functions (UDFs) to PySpark DataFrames.
    
    This class provides methods for creating, registering, and applying UDFs to DataFrames,
    with support for different return types, vectorized UDFs (Pandas UDFs), and UDF caching.
    """
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """
        Initialize the UDF manager.
        
        Args:
            spark_session: SparkSession to use for UDF registration. If None, will attempt
                          to get the active SparkSession.
        """
        if spark_session is None:
            self.spark = SparkSession.getActiveSession()
            if self.spark is None:
                raise ValueError("No active SparkSession found. Please provide a SparkSession.")
        else:
            self.spark = spark_session
        
        # Dictionary to store registered UDFs
        self._udfs = {}
        
        logger.info("Initialized UDFManager")
    
    def register_udf(
        self,
        name: str,
        func: Callable,
        return_type: Any,
        vectorized: bool = False,
        pandas_vectorized: bool = False
    ) -> None:
        """
        Register a User Defined Function (UDF).
        
        Args:
            name: Name to register the UDF under
            func: Python function to register as UDF
            return_type: Return type of the UDF (e.g., StringType(), IntegerType())
            vectorized: Whether to use vectorized UDF (Apache Arrow optimization)
            pandas_vectorized: Whether to use Pandas UDF (requires pandas and pyarrow)
            
        Raises:
            ValueError: If the UDF name is already registered or if invalid parameters are provided
        """
        try:
            if name in self._udfs:
                raise ValueError(f"UDF with name '{name}' is already registered")
            
            if pandas_vectorized and vectorized:
                raise ValueError("Cannot set both vectorized and pandas_vectorized to True")
            
            if pandas_vectorized:
                try:
                    import pandas as pd
                    from pyspark.sql.functions import pandas_udf
                    
                    # Register as Pandas UDF
                    udf_func = pandas_udf(func, return_type)
                    logger.info(f"Registered Pandas UDF: {name}")
                except ImportError:
                    raise ImportError("pandas and pyarrow are required for pandas_vectorized UDFs")
            elif vectorized:
                # Register as vectorized UDF
                udf_func = F.udf(func, return_type, useArrow=True)
                logger.info(f"Registered vectorized UDF: {name}")
            else:
                # Register as regular UDF
                udf_func = F.udf(func, return_type)
                logger.info(f"Registered UDF: {name}")
            
            # Store the UDF in the dictionary
            self._udfs[name] = {
                "function": udf_func,
                "original_func": func,
                "return_type": return_type,
                "vectorized": vectorized,
                "pandas_vectorized": pandas_vectorized
            }
            
            # Also register with Spark SQL if possible
            if not (vectorized or pandas_vectorized):
                self.spark.udf.register(name, udf_func)
                logger.info(f"Registered UDF '{name}' with Spark SQL")
                
        except Exception as e:
            logger.error(f"Failed to register UDF '{name}': {str(e)}")
            raise
    
    def get_udf(self, name: str) -> Optional[Dict]:
        """
        Get a registered UDF by name.
        
        Args:
            name: Name of the UDF to retrieve
            
        Returns:
            Dict containing UDF details if found, None otherwise
        """
        return self._udfs.get(name)
    
    def list_udfs(self) -> List[str]:
        """
        List all registered UDFs.
        
        Returns:
            List of registered UDF names
        """
        return list(self._udfs.keys())
    
    def apply_udf(
        self,
        df: DataFrame,
        udf_name: str,
        input_cols: Union[str, List[str]],
        output_col: str
    ) -> DataFrame:
        """
        Apply a registered UDF to a DataFrame.
        
        Args:
            df: DataFrame to apply the UDF to
            udf_name: Name of the registered UDF to apply
            input_cols: Column(s) to use as input to the UDF
            output_col: Column name for the UDF output
            
        Returns:
            DataFrame with the UDF applied
            
        Raises:
            ValueError: If the UDF is not registered or if invalid parameters are provided
        """
        try:
            if udf_name not in self._udfs:
                raise ValueError(f"UDF '{udf_name}' is not registered")
            
            udf_info = self._udfs[udf_name]
            udf_func = udf_info["function"]
            
            # Convert single column to list for consistent handling
            if isinstance(input_cols, str):
                input_cols = [input_cols]
            
            # Apply the UDF to the DataFrame
            input_cols_expr = [F.col(col) for col in input_cols]
            result_df = df.withColumn(output_col, udf_func(*input_cols_expr))
            
            logger.info(f"Applied UDF '{udf_name}' to create column '{output_col}'")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to apply UDF '{udf_name}': {str(e)}")
            raise
    
    def apply_sql_udf(
        self,
        df: DataFrame,
        udf_name: str,
        sql_expr: str
    ) -> DataFrame:
        """
        Apply a registered UDF using a SQL expression.
        
        Args:
            df: DataFrame to apply the UDF to
            udf_name: Name of the registered UDF
            sql_expr: SQL expression using the UDF (e.g., "udf_name(col1, col2) as new_col")
            
        Returns:
            DataFrame with the SQL expression applied
            
        Raises:
            ValueError: If the UDF is not registered or if invalid parameters are provided
        """
        try:
            if udf_name not in self._udfs:
                raise ValueError(f"UDF '{udf_name}' is not registered")
            
            udf_info = self._udfs[udf_name]
            if udf_info["vectorized"] or udf_info["pandas_vectorized"]:
                raise ValueError("SQL expressions can only be used with regular UDFs")
            
            # Create a temporary view for the DataFrame
            temp_view_name = f"temp_view_{udf_name}_{hash(df)}"
            df.createOrReplaceTempView(temp_view_name)
            
            # Execute the SQL query
            result_df = self.spark.sql(f"SELECT *, {sql_expr} FROM {temp_view_name}")
            
            # Drop the temporary view
            self.spark.catalog.dropTempView(temp_view_name)
            
            logger.info(f"Applied SQL UDF expression: {sql_expr}")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to apply SQL UDF '{udf_name}': {str(e)}")
            raise
    
    def batch_apply_udfs(
        self,
        df: DataFrame,
        udf_configs: List[Dict[str, Any]]
    ) -> DataFrame:
        """
        Apply multiple UDFs to a DataFrame in a single pass.
        
        Args:
            df: DataFrame to apply the UDFs to
            udf_configs: List of dictionaries, each containing:
                        - 'udf_name': Name of the registered UDF
                        - 'input_cols': Column(s) to use as input
                        - 'output_col': Column name for the output
            
        Returns:
            DataFrame with all UDFs applied
            
        Raises:
            ValueError: If any UDF is not registered or if invalid parameters are provided
        """
        try:
            result_df = df
            
            for config in udf_configs:
                udf_name = config.get('udf_name')
                input_cols = config.get('input_cols')
                output_col = config.get('output_col')
                
                if not all([udf_name, input_cols, output_col]):
                    raise ValueError(f"Invalid UDF configuration: {config}")
                
                result_df = self.apply_udf(
                    df=result_df,
                    udf_name=udf_name,
                    input_cols=input_cols,
                    output_col=output_col
                )
            
            logger.info(f"Applied batch of {len(udf_configs)} UDFs")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to apply batch UDFs: {str(e)}")
            raise
    
    def unregister_udf(self, name: str) -> bool:
        """
        Unregister a UDF.
        
        Args:
            name: Name of the UDF to unregister
            
        Returns:
            True if the UDF was unregistered, False if it wasn't registered
        """
        if name in self._udfs:
            del self._udfs[name]
            logger.info(f"Unregistered UDF: {name}")
            return True
        return False
    
    def register_vectorized_pandas_udf(
        self,
        name: str,
        func: Callable,
        return_type: Any,
        input_types: Optional[List[Any]] = None
    ) -> None:
        """
        Register a vectorized Pandas UDF for high-performance operations.
        
        This method registers a Pandas UDF that operates on batches of data using
        pandas Series or DataFrames, which can significantly improve performance
        for computationally intensive operations.
        
        Args:
            name: Name to register the UDF under
            func: Python function that takes pandas Series/DataFrame(s) and returns a pandas Series
            return_type: Return type of the UDF
            input_types: Optional list of input types (for documentation purposes)
            
        Raises:
            ImportError: If pandas or pyarrow are not installed
            ValueError: If the UDF name is already registered
        """
        try:
            # Check if pandas and pyarrow are available
            try:
                import pandas as pd
                import pyarrow
                from pyspark.sql.functions import pandas_udf
            except ImportError:
                raise ImportError("pandas and pyarrow are required for vectorized Pandas UDFs")
            
            if name in self._udfs:
                raise ValueError(f"UDF with name '{name}' is already registered")
            
            # Register the Pandas UDF
            udf_func = pandas_udf(func, return_type)
            
            # Store the UDF in the dictionary
            self._udfs[name] = {
                "function": udf_func,
                "original_func": func,
                "return_type": return_type,
                "vectorized": False,
                "pandas_vectorized": True,
                "input_types": input_types
            }
            
            logger.info(f"Registered vectorized Pandas UDF: {name}")
            
        except Exception as e:
            logger.error(f"Failed to register vectorized Pandas UDF '{name}': {str(e)}")
            raise
    
    def apply_vectorized_udf(
        self,
        df: DataFrame,
        udf_name: str,
        input_cols: Union[str, List[str]],
        output_col: str,
        batch_size: Optional[int] = None
    ) -> DataFrame:
        """
        Apply a vectorized UDF to a DataFrame with optimized performance.
        
        This method applies a vectorized UDF (either Arrow-optimized or Pandas UDF)
        to a DataFrame, with options to control batch size for better performance.
        
        Args:
            df: DataFrame to apply the UDF to
            udf_name: Name of the registered vectorized UDF
            input_cols: Column(s) to use as input
            output_col: Column name for the output
            batch_size: Optional batch size for processing (if supported)
            
        Returns:
            DataFrame with the vectorized UDF applied
            
        Raises:
            ValueError: If the UDF is not registered or is not vectorized
        """
        try:
            if udf_name not in self._udfs:
                raise ValueError(f"UDF '{udf_name}' is not registered")
            
            udf_info = self._udfs[udf_name]
            if not (udf_info["vectorized"] or udf_info["pandas_vectorized"]):
                raise ValueError(f"UDF '{udf_name}' is not a vectorized UDF")
            
            udf_func = udf_info["function"]
            
            # Convert single column to list for consistent handling
            if isinstance(input_cols, str):
                input_cols = [input_cols]
            
            # Set batch size if provided and supported
            if batch_size is not None and udf_info["pandas_vectorized"]:
                # Note: This is a placeholder. In a real implementation, you would
                # need to check if the Spark version supports setting batch size
                # and use the appropriate API.
                logger.warning("Batch size setting is not fully implemented in this version")
            
            # Apply the vectorized UDF
            input_cols_expr = [F.col(col) for col in input_cols]
            result_df = df.withColumn(output_col, udf_func(*input_cols_expr))
            
            logger.info(f"Applied vectorized UDF '{udf_name}' to create column '{output_col}'")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to apply vectorized UDF '{udf_name}': {str(e)}")
            raise


# Example usage
def main():
    """Example usage of the DataFrame operations module."""
    from pyspark.sql import SparkSession
    from pyspark.sql.types import IntegerType, StringType, DoubleType
    
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("DataFrameOpsExample") \
        .getOrCreate()
    
    try:
        # Create a sample DataFrame
        data = [
            (1, "John", 30, "Sales", 5000),
            (2, "Jane", 25, "Engineering", 6000),
            (3, "Bob", 40, "Sales", 4500),
            (4, "Alice", 35, "Engineering", 7000),
            (5, "Charlie", 45, "Marketing", 5500)
        ]
        df = spark.createDataFrame(data, ["id", "name", "age", "department", "salary"])
        
        # Create a transformer
        transformer = DataFrameTransformer(df)
        
        # Example 1: Filter by condition
        filtered_df = transformer.filter_by_condition("age > 30")
        print("Filtered DataFrame (age > 30):")
        filtered_df.show()
        
        # Example 2: Aggregate by department
        agg_df = transformer.aggregate_by(
            group_cols=["department"],
            agg_exprs={"age": "avg", "salary": "sum"}
        )
        print("Aggregated DataFrame by department:")
        agg_df.show()
        
        # Example 3: Apply window function
        window_df = transformer.apply_window_function(
            partition_cols=["department"],
            order_cols=["salary"],
            window_exprs={
                "rank": "rank()",
                "avg_dept_salary": "avg(salary) over (partition by department)"
            }
        )
        print("DataFrame with window functions:")
        window_df.show()
        
        # Example 4: Join with another DataFrame
        other_data = [
            ("Sales", "New York"),
            ("Engineering", "San Francisco"),
            ("Marketing", "Chicago")
        ]
        other_df = spark.createDataFrame(other_data, ["department", "location"])
        
        joined_df = transformer.join_with(
            other_df=other_df,
            join_cols="department",
            join_type="inner"
        )
        print("Joined DataFrame:")
        joined_df.show()
        
        # Example 5: Cache with a specific storage level
        print("\n--- Caching and Optimization Examples ---")
        cached_df = transformer.cache_with_strategy(storage_level="MEMORY_AND_DISK")
        print("Cached DataFrame (check Spark UI for storage):")
        cached_df.show(2)  # Show just 2 rows to demonstrate it's working
        
        # Example 6: Optimize query execution
        optimized_df = transformer.optimize_query_execution(
            enable_adaptive_execution=True,
            enable_broadcast_join=True,
            auto_broadcast_join_threshold=10485760  # 10MB
        )
        print("Optimized DataFrame (query execution):")
        optimized_df.show(2)  # Show just 2 rows
        
        # Example 7: Repartition optimization
        # Create a larger DataFrame for this example
        larger_data = [(i, f"Name-{i}", i % 50, ["Sales", "Engineering", "Marketing"][i % 3], i * 1000) 
                      for i in range(1000)]
        larger_df = spark.createDataFrame(larger_data, ["id", "name", "age", "department", "salary"])
        larger_transformer = DataFrameTransformer(larger_df)
        
        # Repartition by department for better join performance
        repartitioned_df = larger_transformer.repartition_optimize(
            num_partitions=4,
            partition_cols=["department"]
        )
        print("Repartitioned DataFrame:")
        print(f"Number of partitions: {repartitioned_df.rdd.getNumPartitions()}")
        repartitioned_df.show(2)
        
        # Example 8: Coalesce to reduce partitions
        coalesced_df = larger_transformer.repartition_optimize(
            num_partitions=2,
            coalesce_only=True
        )
        print("Coalesced DataFrame:")
        print(f"Number of partitions: {coalesced_df.rdd.getNumPartitions()}")
        coalesced_df.show(2)
        
        # Example 9: UDF Manager
        print("\n--- UDF Manager Examples ---")
        
        # Create a UDF manager
        udf_manager = UDFManager(spark)
        
        # Define some UDFs
        def double_salary(salary):
            """Double the salary value"""
            return salary * 2
        
        def combine_name_dept(name, dept):
            """Combine name and department"""
            return f"{name} ({dept})"
        
        def calculate_bonus(salary, age):
            """Calculate bonus based on salary and age"""
            return salary * (age / 100)
        
        # Register the UDFs
        udf_manager.register_udf(
            name="double_salary",
            func=double_salary,
            return_type=IntegerType()
        )
        
        udf_manager.register_udf(
            name="combine_name_dept",
            func=combine_name_dept,
            return_type=StringType()
        )
        
        udf_manager.register_udf(
            name="calculate_bonus",
            func=calculate_bonus,
            return_type=DoubleType()
        )
        
        # List registered UDFs
        print(f"Registered UDFs: {udf_manager.list_udfs()}")
        
        # Apply a single UDF
        df_with_double_salary = udf_manager.apply_udf(
            df=df,
            udf_name="double_salary",
            input_cols="salary",
            output_col="doubled_salary"
        )
        print("DataFrame with doubled salary:")
        df_with_double_salary.show()
        
        # Apply a UDF with multiple input columns
        df_with_combined_name = udf_manager.apply_udf(
            df=df,
            udf_name="combine_name_dept",
            input_cols=["name", "department"],
            output_col="name_with_dept"
        )
        print("DataFrame with combined name and department:")
        df_with_combined_name.show()
        
        # Apply a UDF using SQL expression
        df_with_bonus = udf_manager.apply_sql_udf(
            df=df,
            udf_name="calculate_bonus",
            sql_expr="calculate_bonus(salary, age) as bonus"
        )
        print("DataFrame with bonus calculated using SQL UDF:")
        df_with_bonus.show()
        
        # Apply multiple UDFs in batch
        batch_config = [
            {
                'udf_name': 'double_salary',
                'input_cols': 'salary',
                'output_col': 'doubled_salary'
            },
            {
                'udf_name': 'combine_name_dept',
                'input_cols': ['name', 'department'],
                'output_col': 'name_with_dept'
            }
        ]
        
        df_with_multiple_udfs = udf_manager.batch_apply_udfs(
            df=df,
            udf_configs=batch_config
        )
        print("DataFrame with multiple UDFs applied:")
        df_with_multiple_udfs.show()
        
        # Example 10: Using UDFs directly from DataFrameTransformer
        print("\n--- DataFrameTransformer with UDFs Examples ---")
        
        # Apply a UDF transformation directly from the transformer
        transformer_with_udf = transformer.apply_udf_transformation(
            udf_manager=udf_manager,
            udf_name="double_salary",
            input_cols="salary",
            output_col="doubled_salary"
        )
        print("Transformer with UDF applied:")
        transformer_with_udf.show()
        
        # Apply a UDF using SQL expression from the transformer
        transformer_with_sql_udf = transformer.apply_udf_transformation(
            udf_manager=udf_manager,
            udf_name="calculate_bonus",
            input_cols=None,  # Not used in SQL mode
            output_col=None,  # Not used in SQL mode
            use_sql=True,
            sql_expr="calculate_bonus(salary, age) as bonus"
        )
        print("Transformer with SQL UDF applied:")
        transformer_with_sql_udf.show()
        
        # Apply multiple UDFs from the transformer
        transformer_with_multiple_udfs = transformer.apply_multiple_udfs(
            udf_manager=udf_manager,
            udf_configs=batch_config
        )
        print("Transformer with multiple UDFs applied:")
        transformer_with_multiple_udfs.show()
        
        # Example 11: Vectorized Pandas UDFs for performance
        print("\n--- Vectorized Pandas UDF Examples ---")
        
        try:
            # This example requires pandas and pyarrow to be installed
            import pandas as pd
            import numpy as np
            from pyspark.sql.types import DoubleType, ArrayType
            
            # Define a vectorized pandas UDF for complex calculation
            def complex_salary_analysis(salary_series, age_series):
                """Calculate a complex salary metric using pandas operations"""
                # This is much more efficient than row-by-row processing
                result = np.log1p(salary_series) * np.sqrt(age_series) / 10
                return pd.Series(result)
            
            # Register the vectorized pandas UDF
            udf_manager.register_vectorized_pandas_udf(
                name="complex_salary_analysis",
                func=complex_salary_analysis,
                return_type=DoubleType()
            )
            
            # Apply the vectorized pandas UDF
            df_with_vectorized_udf = udf_manager.apply_vectorized_udf(
                df=df,
                udf_name="complex_salary_analysis",
                input_cols=["salary", "age"],
                output_col="salary_score"
            )
            
            print("DataFrame with vectorized pandas UDF applied:")
            df_with_vectorized_udf.show()
            
            # Define a pandas UDF that returns an array
            def salary_percentiles(salary_series):
                """Calculate percentiles of salary"""
                percentiles = [25, 50, 75, 90]
                result = [np.percentile(salary_series, p) for p in percentiles]
                return pd.Series([result] * len(salary_series))
            
            # Register the array-returning pandas UDF
            udf_manager.register_vectorized_pandas_udf(
                name="salary_percentiles",
                func=salary_percentiles,
                return_type=ArrayType(DoubleType())
            )
            
            # Apply the array-returning pandas UDF
            df_with_array_udf = udf_manager.apply_vectorized_udf(
                df=larger_df,  # Use the larger DataFrame for this example
                udf_name="salary_percentiles",
                input_cols="salary",
                output_col="salary_percentiles"
            )
            
            print("DataFrame with array-returning pandas UDF applied:")
            df_with_array_udf.select("id", "salary", "salary_percentiles").show(5)
            
        except ImportError:
            print("Pandas and/or PyArrow not installed. Skipping vectorized UDF examples.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Stop the SparkSession
        spark.stop()

# UDFManager Class:
# A dedicated class for managing UDFs with methods for registration, application, and management
# Support for regular UDFs, vectorized UDFs, and Pandas UDFs
# Methods for applying UDFs to DataFrames in various ways (single, batch, SQL)
# Integration with DataFrameTransformer:
# Added methods to the DataFrameTransformer class to work directly with the UDFManager
# Support for applying UDFs as part of a transformation pipeline
# Vectorized UDF Support:
# Added specialized methods for high-performance vectorized UDFs using Pandas
# Support for complex operations and array-returning UDFs
# Comprehensive Examples:
# Updated the main() function with examples demonstrating all the new functionality
# Included examples of different UDF types and application methods
# These additions provide a powerful framework for working with custom functions in PySpark, allowing users to:
# Register and manage UDFs in a centralized way
# Apply UDFs efficiently to DataFrames
# Use high-performance vectorized UDFs for complex operations
# Integrate UDFs into transformation pipelines

if __name__ == "__main__":
    main() 