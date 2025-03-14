"""
Spark Data Access Module

This module provides a comprehensive data access layer for PySpark applications,
with specialized reader classes for different data sources including Azure Data Lake Storage,
Parquet files, and Snowflake. It implements a consistent interface for reading data
regardless of the source.

Example Usage:
    ```python
    # Reading from Azure Data Lake Storage
    from spark_data_access import DataReaderFactory, DataSourceType
    
    # Get a reader for Azure Data Lake Storage
    reader = DataReaderFactory.get_reader(
        source_type=DataSourceType.AZURE_DATA_LAKE,
        spark_session=spark,
        account_name="mystorageaccount",
        container="mycontainer",
        tenant_id="tenant-id",
        client_id="client-id",
        client_secret="client-secret"
    )
    
    # Read a parquet file from ADLS
    df = reader.read_file(
        file_path="path/to/data.parquet",
        file_format="parquet"
    )
    
    # Reading from local Parquet files
    parquet_reader = DataReaderFactory.get_reader(
        source_type=DataSourceType.PARQUET,
        spark_session=spark
    )
    
    df = parquet_reader.read_file(
        file_path="/path/to/local/data.parquet",
        infer_schema=True,
        predicate_pushdown=True
    )
    
    # Reading from Snowflake
    snowflake_reader = DataReaderFactory.get_reader(
        source_type=DataSourceType.SNOWFLAKE,
        spark_session=spark,
        host="account.snowflakecomputing.com",
        warehouse="compute_wh",
        database="analytics",
        schema="public",
        user="username",
        password="password"
    )
    
    df = snowflake_reader.read_table("customers")
    ```
"""

import abc
import enum
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAccessError(Exception):
    """Exception raised for errors in the data access layer."""
    pass


class DataSourceType(enum.Enum):
    """
    Enum representing different types of data sources.
    
    Attributes:
        AZURE_DATA_LAKE: Azure Data Lake Storage Gen2
        PARQUET: Local or distributed Parquet files
        SNOWFLAKE: Snowflake database
    """
    AZURE_DATA_LAKE = "azure_data_lake"
    PARQUET = "parquet"
    SNOWFLAKE = "snowflake"


class DataReader(abc.ABC):
    """
    Abstract base class defining the interface for all data readers.
    
    This interface ensures a consistent pattern for reading data from different sources.
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize the data reader.
        
        Args:
            spark_session: The SparkSession to use for reading data
        """
        self.spark = spark_session
    
    @abc.abstractmethod
    def read_file(self, file_path: str, **options) -> DataFrame:
        """
        Read data from a file.
        
        Args:
            file_path: Path to the file to read
            **options: Additional options for reading the file
            
        Returns:
            DataFrame: The data read from the file
        """
        pass
    
    @abc.abstractmethod
    def read_table(self, table_name: str, **options) -> DataFrame:
        """
        Read data from a table.
        
        Args:
            table_name: Name of the table to read
            **options: Additional options for reading the table
            
        Returns:
            DataFrame: The data read from the table
        """
        pass
    
    @abc.abstractmethod
    def read_query(self, query: str, **options) -> DataFrame:
        """
        Read data using a query.
        
        Args:
            query: Query to execute
            **options: Additional options for executing the query
            
        Returns:
            DataFrame: The data read from the query
        """
        pass
    
    def configure_spark_session(self) -> None:
        """
        Configure the SparkSession with any necessary settings for this reader.
        
        This method should be called before using the reader to ensure the SparkSession
        is properly configured.
        """
        pass


class AzureDataLakeReader(DataReader):
    """
    Reader for Azure Data Lake Storage Gen2.
    
    This reader provides methods for reading data from Azure Data Lake Storage Gen2
    using service principal authentication.
    """
    
    def __init__(
        self,
        spark_session: SparkSession,
        account_name: str,
        container: str,
        tenant_id: str,
        client_id: str,
        client_secret: str
    ):
        """
        Initialize the Azure Data Lake reader.
        
        Args:
            spark_session: The SparkSession to use for reading data
            account_name: Azure Storage account name
            container: Azure Storage container name
            tenant_id: Azure tenant ID
            client_id: Azure client ID (service principal)
            client_secret: Azure client secret
        """
        super().__init__(spark_session)
        self.account_name = account_name
        self.container = container
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Configure the SparkSession for Azure Data Lake access
        self.configure_spark_session()
    
    def configure_spark_session(self) -> None:
        """
        Configure the SparkSession with Azure Data Lake Storage settings.
        
        This method adds the necessary configurations to the SparkSession to enable
        access to Azure Data Lake Storage Gen2 using service principal authentication.
        """
        try:
            # Set Azure Data Lake Storage configurations
            self.spark.conf.set(
                f"fs.azure.account.auth.type.{self.account_name}.dfs.core.windows.net",
                "OAuth"
            )
            self.spark.conf.set(
                f"fs.azure.account.oauth.provider.type.{self.account_name}.dfs.core.windows.net",
                "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
            )
            self.spark.conf.set(
                f"fs.azure.account.oauth2.client.id.{self.account_name}.dfs.core.windows.net",
                self.client_id
            )
            self.spark.conf.set(
                f"fs.azure.account.oauth2.client.secret.{self.account_name}.dfs.core.windows.net",
                self.client_secret
            )
            self.spark.conf.set(
                f"fs.azure.account.oauth2.client.endpoint.{self.account_name}.dfs.core.windows.net",
                f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"
            )
            
            # Add required JAR files if not already included
            jars = [
                "com.microsoft.azure:azure-storage:8.6.6",
                "org.apache.hadoop:hadoop-azure:3.3.1",
                "com.microsoft.azure:azure-data-lake-store-sdk:2.3.9"
            ]
            
            # Check if jars are already in the classpath
            current_jars = self.spark.sparkContext.getConf().get("spark.jars", "")
            for jar in jars:
                if jar not in current_jars:
                    # Add the jar to the classpath
                    # Note: In a real implementation, you would need to restart the SparkSession
                    # to apply these changes, which is not practical in a running application.
                    # This is just to demonstrate the concept.
                    logger.warning(
                        f"JAR {jar} should be included in the SparkSession configuration. "
                        "In a real implementation, you would need to include this JAR "
                        "when creating the SparkSession."
                    )
            
            logger.info("Configured SparkSession for Azure Data Lake Storage access")
        except Exception as e:
            logger.error(f"Failed to configure SparkSession for Azure Data Lake: {str(e)}")
            raise DataAccessError(f"Azure Data Lake configuration error: {str(e)}")
    
    def read_file(
        self,
        file_path: str,
        file_format: str = "parquet",
        schema: Optional[StructType] = None,
        **options
    ) -> DataFrame:
        """
        Read a file from Azure Data Lake Storage.
        
        Args:
            file_path: Path to the file within the container
            file_format: Format of the file (parquet, csv, json, etc.)
            schema: Optional schema for the DataFrame
            **options: Additional options for reading the file
            
        Returns:
            DataFrame: The data read from the file
        """
        try:
            # Construct the full path to the file
            full_path = f"abfss://{self.container}@{self.account_name}.dfs.core.windows.net/{file_path}"
            
            # Read the file using the appropriate method based on the format
            reader = self.spark.read.options(**options)
            
            if schema:
                reader = reader.schema(schema)
            
            if file_format.lower() == "parquet":
                df = reader.parquet(full_path)
            elif file_format.lower() == "csv":
                df = reader.csv(full_path)
            elif file_format.lower() == "json":
                df = reader.json(full_path)
            elif file_format.lower() == "orc":
                df = reader.orc(full_path)
            elif file_format.lower() == "avro":
                df = reader.format("avro").load(full_path)
            else:
                df = reader.format(file_format).load(full_path)
            
            logger.info(f"Successfully read {file_format} file from {full_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to read file from Azure Data Lake: {str(e)}")
            raise DataAccessError(f"Azure Data Lake read error: {str(e)}")
    
    def read_table(self, table_name: str, **options) -> DataFrame:
        """
        Read a table from Azure Data Lake Storage.
        
        This method is not directly applicable to Azure Data Lake Storage,
        but is implemented to satisfy the interface. It raises an error.
        
        Args:
            table_name: Name of the table to read
            **options: Additional options for reading the table
            
        Returns:
            DataFrame: The data read from the table
            
        Raises:
            DataAccessError: This operation is not supported for Azure Data Lake Storage
        """
        raise DataAccessError(
            "Reading tables directly is not supported for Azure Data Lake Storage. "
            "Use read_file() instead with the appropriate file path."
        )
    
    def read_query(self, query: str, **options) -> DataFrame:
        """
        Execute a query against Azure Data Lake Storage.
        
        This method is not directly applicable to Azure Data Lake Storage,
        but is implemented to satisfy the interface. It raises an error.
        
        Args:
            query: Query to execute
            **options: Additional options for executing the query
            
        Returns:
            DataFrame: The data read from the query
            
        Raises:
            DataAccessError: This operation is not supported for Azure Data Lake Storage
        """
        raise DataAccessError(
            "Executing queries directly is not supported for Azure Data Lake Storage. "
            "Use read_file() instead with the appropriate file path."
        )


class ParquetReader(DataReader):
    """
    Reader for Parquet files.
    
    This reader provides methods for reading Parquet files with options for
    schema inference, partitioning, and predicate pushdown.
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize the Parquet reader.
        
        Args:
            spark_session: The SparkSession to use for reading data
        """
        super().__init__(spark_session)
    
    def read_file(
        self,
        file_path: str,
        schema: Optional[StructType] = None,
        infer_schema: bool = True,
        predicate_pushdown: bool = True,
        **options
    ) -> DataFrame:
        """
        Read a Parquet file.
        
        Args:
            file_path: Path to the Parquet file
            schema: Optional schema for the DataFrame
            infer_schema: Whether to infer the schema from the file
            predicate_pushdown: Whether to enable predicate pushdown
            **options: Additional options for reading the file
            
        Returns:
            DataFrame: The data read from the file
        """
        try:
            # Configure predicate pushdown
            self.spark.conf.set("spark.sql.parquet.filterPushdown", str(predicate_pushdown).lower())
            
            # Configure schema inference
            self.spark.conf.set("spark.sql.parquet.inferSchema", str(infer_schema).lower())
            
            # Read the Parquet file
            reader = self.spark.read.options(**options)
            
            if schema:
                reader = reader.schema(schema)
            
            df = reader.parquet(file_path)
            
            logger.info(f"Successfully read Parquet file from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to read Parquet file: {str(e)}")
            raise DataAccessError(f"Parquet read error: {str(e)}")
    
    def read_table(self, table_name: str, **options) -> DataFrame:
        """
        Read a table from the Spark catalog.
        
        Args:
            table_name: Name of the table to read
            **options: Additional options for reading the table
            
        Returns:
            DataFrame: The data read from the table
        """
        try:
            df = self.spark.table(table_name)
            logger.info(f"Successfully read table {table_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to read table: {str(e)}")
            raise DataAccessError(f"Table read error: {str(e)}")
    
    def read_query(self, query: str, **options) -> DataFrame:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            **options: Additional options for executing the query
            
        Returns:
            DataFrame: The data read from the query
        """
        try:
            df = self.spark.sql(query)
            logger.info(f"Successfully executed query: {query}")
            return df
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            raise DataAccessError(f"Query execution error: {str(e)}")


class SnowflakeReader(DataReader):
    """
    Reader for Snowflake database.
    
    This reader provides methods for reading data from Snowflake with connection
    parameters for host, warehouse, database, schema, and query options.
    """
    
    def __init__(
        self,
        spark_session: SparkSession,
        host: str,
        warehouse: str,
        database: str,
        schema: str,
        user: str,
        password: str = None,
        private_key_path: str = None,
        private_key_passphrase: str = None,
        role: str = None,
        connection_timeout: int = 60
    ):
        """
        Initialize the Snowflake reader.
        
        Args:
            spark_session: The SparkSession to use for reading data
            host: Snowflake account URL (e.g., account.snowflakecomputing.com)
            warehouse: Snowflake warehouse name
            database: Snowflake database name
            schema: Snowflake schema name
            user: Snowflake username
            password: Snowflake password (optional if using private key)
            private_key_path: Path to private key file (optional if using password)
            private_key_passphrase: Passphrase for private key (optional)
            role: Snowflake role (optional)
            connection_timeout: Connection timeout in seconds (default: 60)
        """
        super().__init__(spark_session)
        self.host = host
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.user = user
        self.password = password
        self.private_key_path = private_key_path
        self.private_key_passphrase = private_key_passphrase
        self.role = role
        self.connection_timeout = connection_timeout
        
        # Validate authentication parameters
        if not password and not private_key_path:
            raise DataAccessError(
                "Either password or private_key_path must be provided for Snowflake authentication"
            )
        
        # Configure the SparkSession for Snowflake access
        self.configure_spark_session()
        
        # Connection pool for reusing connections
        self._connection_pool = set()
    
    def configure_spark_session(self) -> None:
        """
        Configure the SparkSession with Snowflake settings.
        
        This method adds the necessary configurations to the SparkSession to enable
        access to Snowflake using the provided credentials.
        """
        try:
            # Add required JAR files if not already included
            jars = [
                "net.snowflake:snowflake-jdbc:3.13.14",
                "net.snowflake:spark-snowflake_2.12:2.10.0-spark_3.2"
            ]
            
            # Check if jars are already in the classpath
            current_jars = self.spark.sparkContext.getConf().get("spark.jars", "")
            for jar in jars:
                if jar not in current_jars:
                    # Add the jar to the classpath
                    # Note: In a real implementation, you would need to restart the SparkSession
                    # to apply these changes, which is not practical in a running application.
                    # This is just to demonstrate the concept.
                    logger.warning(
                        f"JAR {jar} should be included in the SparkSession configuration. "
                        "In a real implementation, you would need to include this JAR "
                        "when creating the SparkSession."
                    )
            
            logger.info("Configured SparkSession for Snowflake access")
        except Exception as e:
            logger.error(f"Failed to configure SparkSession for Snowflake: {str(e)}")
            raise DataAccessError(f"Snowflake configuration error: {str(e)}")
    
    def _get_snowflake_options(self, **additional_options) -> Dict[str, str]:
        """
        Get the options for connecting to Snowflake.
        
        Args:
            **additional_options: Additional options to include
            
        Returns:
            Dict[str, str]: The options for connecting to Snowflake
        """
        options = {
            "sfUrl": self.host,
            "sfUser": self.user,
            "sfDatabase": self.database,
            "sfSchema": self.schema,
            "sfWarehouse": self.warehouse,
            "sfRole": self.role if self.role else "",
            "sfConnectTimeout": str(self.connection_timeout)
        }
        
        # Add authentication options
        if self.password:
            options["sfPassword"] = self.password
        elif self.private_key_path:
            options["pem_private_key"] = self.private_key_path
            if self.private_key_passphrase:
                options["pem_private_key_passphrase"] = self.private_key_passphrase
        
        # Add any additional options
        options.update(additional_options)
        
        return options
    
    def read_file(self, file_path: str, **options) -> DataFrame:
        """
        Read a file from Snowflake.
        
        This method is not directly applicable to Snowflake,
        but is implemented to satisfy the interface. It raises an error.
        
        Args:
            file_path: Path to the file
            **options: Additional options for reading the file
            
        Returns:
            DataFrame: The data read from the file
            
        Raises:
            DataAccessError: This operation is not supported for Snowflake
        """
        raise DataAccessError(
            "Reading files directly is not supported for Snowflake. "
            "Use read_table() or read_query() instead."
        )
    
    def read_table(self, table_name: str, **options) -> DataFrame:
        """
        Read a table from Snowflake.
        
        Args:
            table_name: Name of the table to read
            **options: Additional options for reading the table
            
        Returns:
            DataFrame: The data read from the table
        """
        try:
            # Get Snowflake connection options
            sf_options = self._get_snowflake_options(**options)
            
            # Read the table from Snowflake
            df = self.spark.read \
                .format("snowflake") \
                .options(**sf_options) \
                .option("dbtable", table_name) \
                .load()
            
            logger.info(f"Successfully read table {table_name} from Snowflake")
            return df
        except Exception as e:
            logger.error(f"Failed to read table from Snowflake: {str(e)}")
            raise DataAccessError(f"Snowflake table read error: {str(e)}")
    
    def read_query(self, query: str, **options) -> DataFrame:
        """
        Execute a query against Snowflake.
        
        Args:
            query: SQL query to execute
            **options: Additional options for executing the query
            
        Returns:
            DataFrame: The data read from the query
        """
        try:
            # Get Snowflake connection options
            sf_options = self._get_snowflake_options(**options)
            
            # Execute the query against Snowflake
            df = self.spark.read \
                .format("snowflake") \
                .options(**sf_options) \
                .option("query", query) \
                .load()
            
            logger.info(f"Successfully executed query against Snowflake: {query}")
            return df
        except Exception as e:
            logger.error(f"Failed to execute query against Snowflake: {str(e)}")
            raise DataAccessError(f"Snowflake query execution error: {str(e)}")
    
    def __del__(self):
        """
        Clean up resources when the reader is garbage collected.
        
        This method ensures that any open connections are properly closed.
        """
        self._close_connections()
    
    def _close_connections(self):
        """
        Close all open connections in the connection pool.
        
        This method should be called when the reader is no longer needed
        to ensure proper resource cleanup.
        """
        for connection in self._connection_pool:
            try:
                # In a real implementation, you would close the connection here
                pass
            except Exception as e:
                logger.warning(f"Error closing Snowflake connection: {str(e)}")
        
        self._connection_pool.clear()


class DataReaderFactory:
    """
    Factory for creating data readers.
    
    This factory provides methods for creating the appropriate data reader
    based on the source type and configuration provided.
    """
    
    @staticmethod
    def get_reader(source_type: DataSourceType, spark_session: SparkSession, **kwargs) -> DataReader:
        """
        Get a data reader for the specified source type.
        
        Args:
            source_type: Type of data source
            spark_session: SparkSession to use for reading data
            **kwargs: Additional arguments for the reader
            
        Returns:
            DataReader: The appropriate data reader for the source type
            
        Raises:
            DataAccessError: If the source type is not supported or required parameters are missing
        """
        try:
            if source_type == DataSourceType.AZURE_DATA_LAKE:
                # Check required parameters for Azure Data Lake Storage
                required_params = ["account_name", "container", "tenant_id", "client_id", "client_secret"]
                for param in required_params:
                    if param not in kwargs:
                        raise DataAccessError(f"Missing required parameter for Azure Data Lake Storage: {param}")
                
                return AzureDataLakeReader(
                    spark_session=spark_session,
                    account_name=kwargs["account_name"],
                    container=kwargs["container"],
                    tenant_id=kwargs["tenant_id"],
                    client_id=kwargs["client_id"],
                    client_secret=kwargs["client_secret"]
                )
            
            elif source_type == DataSourceType.PARQUET:
                return ParquetReader(spark_session=spark_session)
            
            elif source_type == DataSourceType.SNOWFLAKE:
                # Check required parameters for Snowflake
                required_params = ["host", "warehouse", "database", "schema", "user"]
                for param in required_params:
                    if param not in kwargs:
                        raise DataAccessError(f"Missing required parameter for Snowflake: {param}")
                
                # Either password or private_key_path must be provided
                if "password" not in kwargs and "private_key_path" not in kwargs:
                    raise DataAccessError(
                        "Either password or private_key_path must be provided for Snowflake authentication"
                    )
                
                return SnowflakeReader(
                    spark_session=spark_session,
                    host=kwargs["host"],
                    warehouse=kwargs["warehouse"],
                    database=kwargs["database"],
                    schema=kwargs["schema"],
                    user=kwargs["user"],
                    password=kwargs.get("password"),
                    private_key_path=kwargs.get("private_key_path"),
                    private_key_passphrase=kwargs.get("private_key_passphrase"),
                    role=kwargs.get("role"),
                    connection_timeout=kwargs.get("connection_timeout", 60)
                )
            
            else:
                raise DataAccessError(f"Unsupported data source type: {source_type}")
        
        except Exception as e:
            if isinstance(e, DataAccessError):
                raise
            else:
                logger.error(f"Failed to create data reader: {str(e)}")
                raise DataAccessError(f"Failed to create data reader: {str(e)}")


# Example usage
def main():
    """Example usage of the data access layer."""
    from pyspark.sql import SparkSession
    
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("DataAccessExample") \
        .getOrCreate()
    
    try:
        # Example 1: Reading from Azure Data Lake Storage
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
        df1 = adls_reader.read_file(
            file_path="path/to/data.parquet",
            file_format="parquet"
        )
        
        # Example 2: Reading from local Parquet files
        parquet_reader = DataReaderFactory.get_reader(
            source_type=DataSourceType.PARQUET,
            spark_session=spark
        )
        
        df2 = parquet_reader.read_file(
            file_path="/path/to/local/data.parquet",
            infer_schema=True,
            predicate_pushdown=True
        )
        
        # Example 3: Reading from Snowflake
        snowflake_reader = DataReaderFactory.get_reader(
            source_type=DataSourceType.SNOWFLAKE,
            spark_session=spark,
            host="account.snowflakecomputing.com",
            warehouse="compute_wh",
            database="analytics",
            schema="public",
            user="username",
            password="password"
        )
        
        df3 = snowflake_reader.read_table("customers")
        
        # Show the results
        print("Data from Azure Data Lake Storage:")
        df1.show(5)
        
        print("Data from local Parquet file:")
        df2.show(5)
        
        print("Data from Snowflake:")
        df3.show(5)
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Stop the SparkSession
        spark.stop()


if __name__ == "__main__":
    main() 