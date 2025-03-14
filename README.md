# PySpark Utilities

A comprehensive collection of utility classes for working with Apache Spark in Python. This library provides abstractions for common Spark operations, including DataFrame transformations, data access patterns, UDF management, and Kubernetes deployment.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [DataFrameTransformer](#dataframetransformer)
  - [UDFManager](#udfmanager)
  - [KubernetesSparkJob](#kubernetesspark-job)
  - [DataReaderFactory](#datareaderfactory)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Examples](#examples)
  - [DataFrame Transformations](#dataframe-transformations)
  - [Custom UDFs](#custom-udfs)
  - [Reading from Azure Data Lake](#reading-from-azure-data-lake)
  - [Running a Spark Job on Kubernetes](#running-a-spark-job-on-kubernetes)
- [Advanced Usage](#advanced-usage)
  - [Caching and Persistence](#caching-and-persistence)
  - [Query Optimization](#query-optimization)
  - [Vectorized UDFs](#vectorized-udfs)
- [Contributing](#contributing)

## Overview

This library provides a set of utility classes that make working with Apache Spark in Python more efficient and maintainable. It includes abstractions for common operations such as DataFrame transformations, data access, UDF management, and Kubernetes deployment.

The main goal is to provide a consistent and easy-to-use interface for Spark operations, reducing boilerplate code and making Spark applications more maintainable.

## Components

### DataFrameTransformer

The `DataFrameTransformer` class provides a fluent interface for applying transformations to Spark DataFrames. It includes methods for filtering, joining, aggregating, and applying window functions, as well as optimizations like caching and repartitioning.

Key features:
- Filtering and selection operations
- Join operations
- Aggregation and window functions
- Column operations (renaming, casting, etc.)
- Caching and persistence strategies
- Query optimization
- UDF integration

### UDFManager

The `UDFManager` class provides a centralized way to manage User Defined Functions (UDFs) in Spark applications. It includes methods for registering, applying, and managing UDFs, with support for different return types, vectorized UDFs, and Pandas UDFs.

Key features:
- Registration of regular, vectorized, and Pandas UDFs
- Application of UDFs to DataFrames
- SQL-based UDF application
- Batch application of multiple UDFs
- High-performance vectorized UDFs

### KubernetesSparkJob

The `KubernetesSparkJob` class provides an abstraction for running Spark jobs on Kubernetes. It handles the configuration, submission, and management of Spark jobs in a Kubernetes environment.

Key features:
- Configuration of Spark on Kubernetes
- Job submission and management
- Context manager for SparkSession lifecycle
- Cluster size configuration

### DataReaderFactory

The `DataReaderFactory` class provides a factory for creating data readers for different data sources. It includes readers for Azure Data Lake Storage, Parquet files, and Snowflake.

Key features:
- Unified interface for different data sources
- Specialized readers for Azure Data Lake Storage, Parquet, and Snowflake
- Configuration of data source connections
- Reading files, tables, and executing queries

## Getting Started

### Installation

To use these utilities, you need to have Apache Spark installed. You can install the required dependencies using pip:

```bash
pip install pyspark==3.3.0
```

For specific components, you may need additional dependencies:

```bash
# For Azure Data Lake Storage
pip install azure-storage-blob azure-identity

# For Snowflake
pip install snowflake-connector-python

# For Pandas UDFs
pip install pandas pyarrow
```

### Basic Usage

Here's a simple example of using the `DataFrameTransformer` to transform a DataFrame:

```python
from pyspark.sql import SparkSession
from spark_dataframe_ops import DataFrameTransformer

# Create a SparkSession
spark = SparkSession.builder \
    .appName("SimpleExample") \
    .getOrCreate()

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

# Stop the SparkSession
spark.stop()
```

## Examples

### DataFrame Transformations

Here's an example of using various transformations with the `DataFrameTransformer`:

```python
from pyspark.sql import SparkSession
from spark_dataframe_ops import DataFrameTransformer

# Create a SparkSession
spark = SparkSession.builder \
    .appName("TransformationsExample") \
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
    
    # Filter by condition
    filtered_df = transformer.filter_by_condition("age > 30")
    print("Filtered DataFrame (age > 30):")
    filtered_df.show()
    
    # Aggregate by department
    agg_df = transformer.aggregate_by(
        group_cols=["department"],
        agg_exprs={"age": "avg", "salary": "sum"}
    )
    print("Aggregated DataFrame by department:")
    agg_df.show()
    
    # Apply window function
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
    
    # Join with another DataFrame
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
    
finally:
    # Stop the SparkSession
    spark.stop()
```

### Custom UDFs

Here's an example of using the `UDFManager` to register and apply custom UDFs:

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, DoubleType
from spark_dataframe_ops import DataFrameTransformer, UDFManager

# Create a SparkSession
spark = SparkSession.builder \
    .appName("UDFExample") \
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
    
finally:
    # Stop the SparkSession
    spark.stop()
```

### Reading from Azure Data Lake

Here's an example of using the `DataReaderFactory` to read data from Azure Data Lake Storage:

```python
from pyspark.sql import SparkSession
from spark_data_access import DataReaderFactory, DataSourceType

# Create a SparkSession
spark = SparkSession.builder \
    .appName("AzureDataLakeExample") \
    .getOrCreate()

try:
    # Get a reader for Azure Data Lake Storage
    adls_reader = DataReaderFactory.get_reader(
        source_type=DataSourceType.AZURE_DATA_LAKE,
        spark_session=spark,
        account_name="mystorageaccount",
        container="mycontainer",
        tenant_id="tenant-id",
        client_id="client-id",
        client_secret="client-secret"
    )
    
    # Read a parquet file from ADLS
    df = adls_reader.read_file(
        file_path="path/to/data.parquet",
        file_format="parquet"
    )
    
    # Show the data
    print("Data from Azure Data Lake Storage:")
    df.show(5)
    
    # Apply transformations using DataFrameTransformer
    transformer = DataFrameTransformer(df)
    
    # Filter and aggregate the data
    filtered_df = transformer.filter_by_condition("column_name > 100")
    aggregated_df = transformer.aggregate_by(
        group_cols=["category_column"],
        agg_exprs={"value_column": "sum"}
    )
    
    # Show the results
    print("Filtered data:")
    filtered_df.show(5)
    
    print("Aggregated data:")
    aggregated_df.show(5)
    
finally:
    # Stop the SparkSession
    spark.stop()
```

### Running a Spark Job on Kubernetes

Here's an example of using the `KubernetesSparkJob` to run a Spark job on Kubernetes:

```python
from pyspark.sql import SparkSession
from spark_wrapper import KubernetesSparkJob, ClusterSize, SparkJobConfig

# Define a job function
def my_spark_job(spark):
    # Create a simple DataFrame
    data = [("Alice", 34), ("Bob", 45), ("Charlie", 29)]
    df = spark.createDataFrame(data, ["Name", "Age"])
    
    # Perform some transformations
    result = df.groupBy().avg("Age").collect()[0][0]
    
    return result

# Configure the job
config = SparkJobConfig(
    cluster_size=ClusterSize.SMALL,
    master_address="kubernetes.default.svc",
    namespace="spark-jobs",
    image="apache/spark:3.3.0",
    driver_memory="2g",
    executor_memory="2g",
    executor_cores=1
)

# Run the job using the context manager
try:
    with KubernetesSparkJob(app_name="example-job", config=config) as job:
        avg_age = job.submit(my_spark_job)
        print(f"Average age: {avg_age}")
except Exception as e:
    print(f"Job failed: {str(e)}")
```

## Advanced Usage

### Caching and Persistence

The `DataFrameTransformer` provides methods for caching DataFrames with different persistence levels:

```python
from pyspark.sql import SparkSession
from spark_dataframe_ops import DataFrameTransformer

# Create a SparkSession
spark = SparkSession.builder \
    .appName("CachingExample") \
    .getOrCreate()

try:
    # Create a sample DataFrame
    data = [(i, f"value_{i}") for i in range(1000)]
    df = spark.createDataFrame(data, ["id", "value"])
    
    # Create a transformer
    transformer = DataFrameTransformer(df)
    
    # Cache with a specific storage level
    cached_df = transformer.cache_with_strategy(storage_level="MEMORY_AND_DISK")
    
    # Perform operations on the cached DataFrame
    result = cached_df.count()
    print(f"Count: {result}")
    
finally:
    # Stop the SparkSession
    spark.stop()
```

### Query Optimization

The `DataFrameTransformer` provides methods for optimizing query execution:

```python
from pyspark.sql import SparkSession
from spark_dataframe_ops import DataFrameTransformer

# Create a SparkSession
spark = SparkSession.builder \
    .appName("OptimizationExample") \
    .getOrCreate()

try:
    # Create sample DataFrames
    data1 = [(i, f"value_{i}") for i in range(1000)]
    data2 = [(i, f"category_{i % 5}") for i in range(1000)]
    
    df1 = spark.createDataFrame(data1, ["id", "value"])
    df2 = spark.createDataFrame(data2, ["id", "category"])
    
    # Create transformers
    transformer1 = DataFrameTransformer(df1)
    
    # Optimize query execution
    optimized_df = transformer1.optimize_query_execution(
        enable_adaptive_execution=True,
        enable_broadcast_join=True,
        auto_broadcast_join_threshold=10485760  # 10MB
    )
    
    # Join with the optimized settings
    joined_df = optimized_df.join(df2, "id")
    
    # Show the result
    joined_df.show(5)
    
    # Optimize partitioning
    repartitioned_df = transformer1.repartition_optimize(
        num_partitions=4,
        partition_cols=["id"]
    )
    
    print(f"Number of partitions: {repartitioned_df.rdd.getNumPartitions()}")
    
finally:
    # Stop the SparkSession
    spark.stop()
```

### Vectorized UDFs

The `UDFManager` provides support for high-performance vectorized UDFs using Pandas:

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, ArrayType
from spark_dataframe_ops import DataFrameTransformer, UDFManager

# Create a SparkSession
spark = SparkSession.builder \
    .appName("VectorizedUDFExample") \
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
    
    # Create a UDF manager
    udf_manager = UDFManager(spark)
    
    # Import required libraries
    import pandas as pd
    import numpy as np
    
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
    
finally:
    # Stop the SparkSession
    spark.stop()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 