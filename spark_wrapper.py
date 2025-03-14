"""
Spark Wrapper Module

This module provides a flexible abstraction for running PySpark jobs on Kubernetes
with configurable cluster sizes. It handles the configuration, submission, and
management of Spark jobs in a Kubernetes environment.

Example Usage:
    ```python
    # Basic usage with default configuration
    from spark_wrapper import KubernetesSparkJob, ClusterSize
    
    # Define a job function
    def my_spark_job(spark):
        df = spark.read.csv("s3://my-bucket/data.csv")
        result = df.groupBy("column").count()
        return result
    
    # Run the job with a medium-sized cluster
    with KubernetesSparkJob(
        cluster_size=ClusterSize.MEDIUM,
        app_name="my-analytics-job",
        namespace="data-processing"
    ) as job:
        result_df = job.submit(my_spark_job)
        result_df.show()
    
    # Advanced usage with custom configuration
    from spark_wrapper import KubernetesSparkJob, SparkJobConfig
    
    config = SparkJobConfig(
        master_address="k8s-api.example.com:443",
        driver_memory="4g",
        executor_memory="8g",
        executor_cores=2,
        executor_instances=6,
        image="my-registry/spark:3.3.0"
    )
    
    with KubernetesSparkJob(
        config=config,
        app_name="advanced-analytics",
        namespace="data-science"
    ) as job:
        result = job.submit(my_spark_job)
    ```
"""

import abc
import enum
import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from pyspark import SparkConf
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
SparkJobFunction = Callable[[SparkSession], T]


class ValidationError(Exception):
    """Exception raised for validation errors in the Spark job configuration."""
    pass


class ClusterSize(enum.Enum):
    """
    Enum representing different cluster sizes for Spark jobs.
    
    Attributes:
        SMALL: Small cluster with 2 executors
        MEDIUM: Medium cluster with 6 executors
        LARGE: Large cluster with 10 executors
    """
    SMALL = 2
    MEDIUM = 6
    LARGE = 10


def validate_cluster_size(func):
    """
    Decorator to validate that the cluster size is a valid ClusterSize enum value.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
        
    Raises:
        ValidationError: If the cluster size is invalid
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract cluster_size from args or kwargs
        cluster_size = None
        
        # Check if cluster_size is in kwargs
        if 'cluster_size' in kwargs:
            cluster_size = kwargs['cluster_size']
        # Check if it's in args (assuming it's the second parameter after self)
        elif len(args) > 1:
            cluster_size = args[1]
        # If we're using a config object, check that
        elif 'config' in kwargs and hasattr(kwargs['config'], 'cluster_size'):
            cluster_size = kwargs['config'].cluster_size
        elif len(args) > 1 and hasattr(args[1], 'cluster_size'):
            cluster_size = args[1].cluster_size
            
        # Validate the cluster size
        if cluster_size is not None and not isinstance(cluster_size, ClusterSize):
            valid_sizes = [size.name for size in ClusterSize]
            raise ValidationError(
                f"Invalid cluster size: {cluster_size}. "
                f"Must be one of: {', '.join(valid_sizes)}"
            )
            
        return func(*args, **kwargs)
    
    return wrapper


@dataclass
class SparkJobConfig:
    """
    Configuration class for Spark jobs.
    
    Attributes:
        cluster_size: Size of the cluster (SMALL, MEDIUM, LARGE)
        master_address: Kubernetes API server address
        namespace: Kubernetes namespace
        image: Container image for Spark executors
        driver_memory: Memory allocation for the driver
        executor_memory: Memory allocation for each executor
        executor_cores: Number of cores for each executor
        executor_instances: Number of executor instances
        service_account: Kubernetes service account for Spark
        driver_bind_address: Bind address for the driver
        driver_host: Hostname for the driver service
        driver_port: Port for the driver service
        block_manager_port: Port for the block manager
        image_pull_policy: Kubernetes image pull policy
        extra_conf: Additional Spark configuration parameters
    """
    cluster_size: ClusterSize = ClusterSize.MEDIUM
    master_address: str = ""
    namespace: str = "default"
    image: str = ""
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 2
    executor_instances: Optional[int] = None
    service_account: str = "spark"
    driver_bind_address: str = "0.0.0.0"
    driver_host: str = ""
    driver_port: str = "2222"
    block_manager_port: str = "7777"
    image_pull_policy: str = "IfNotPresent"
    extra_conf: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived values after initialization."""
        # Set executor instances based on cluster size if not explicitly provided
        if self.executor_instances is None:
            self.executor_instances = self.cluster_size.value


class SparkJobInterface(abc.ABC):
    """
    Abstract base class defining the interface for Spark job implementations.
    
    This interface defines the contract that all Spark job implementations must follow,
    including methods for job submission and configuration validation.
    """
    
    @abc.abstractmethod
    def validate_configuration(self) -> bool:
        """
        Validate the job configuration.
        
        Returns:
            bool: True if the configuration is valid, False otherwise
        
        Raises:
            ValidationError: If the configuration is invalid
        """
        pass
    
    @abc.abstractmethod
    def submit(self, job_function: SparkJobFunction[T], *args: Any, **kwargs: Any) -> T:
        """
        Submit a Spark job for execution.
        
        Args:
            job_function: Function that takes a SparkSession and returns a result
            *args: Additional positional arguments to pass to the job function
            **kwargs: Additional keyword arguments to pass to the job function
            
        Returns:
            The result of the job function
        """
        pass
    
    @abc.abstractmethod
    def create_spark_session(self) -> SparkSession:
        """
        Create a SparkSession with the appropriate configuration.
        
        Returns:
            SparkSession: Configured Spark session
        """
        pass


class KubernetesSparkJob(SparkJobInterface):
    """
    Implementation of SparkJobInterface for Kubernetes-based Spark clusters.
    
    This class handles the configuration, submission, and management of Spark jobs
    in a Kubernetes environment.
    """
    
    def __init__(
        self,
        app_name: str,
        config: Optional[SparkJobConfig] = None,
        cluster_size: ClusterSize = ClusterSize.MEDIUM,
        namespace: str = "default",
        master_address: str = "",
        image: str = "",
    ):
        """
        Initialize a new KubernetesSparkJob.
        
        Args:
            app_name: Name of the Spark application
            config: Optional SparkJobConfig object
            cluster_size: Size of the cluster (SMALL, MEDIUM, LARGE)
            namespace: Kubernetes namespace
            master_address: Kubernetes API server address
            image: Container image for Spark executors
        """
        self.app_name = app_name
        
        # Use provided config or create a new one
        if config is not None:
            self.config = config
        else:
            self.config = SparkJobConfig(
                cluster_size=cluster_size,
                namespace=namespace,
                master_address=master_address,
                image=image,
            )
        
        self.spark: Optional[SparkSession] = None
        logger.info(f"Initialized KubernetesSparkJob with app_name={app_name}")
    
    def validate_configuration(self) -> bool:
        """
        Validate the job configuration.
        
        Returns:
            bool: True if the configuration is valid
            
        Raises:
            ValidationError: If the configuration is invalid
        """
        # Check required fields
        if not self.app_name:
            raise ValidationError("Application name is required")
        
        if not self.config.master_address:
            raise ValidationError("Kubernetes master address is required")
        
        if not self.config.image:
            raise ValidationError("Container image is required")
        
        # Validate cluster size
        if not isinstance(self.config.cluster_size, ClusterSize):
            valid_sizes = [size.name for size in ClusterSize]
            raise ValidationError(
                f"Invalid cluster size: {self.config.cluster_size}. "
                f"Must be one of: {', '.join(valid_sizes)}"
            )
        
        logger.info("Configuration validation successful")
        return True
    
    def create_spark_conf(self) -> SparkConf:
        """
        Create a SparkConf with the appropriate Kubernetes configuration.
        
        Returns:
            SparkConf: Configured SparkConf object
        """
        conf = SparkConf()
        
        # Set master URL
        conf.setMaster(f"k8s://{self.config.master_address}")
        
        # Set basic configuration
        conf.set("spark.app.name", self.app_name)
        conf.set("spark.submit.deployMode", "client")
        
        # Set Kubernetes-specific configuration
        conf.set("spark.kubernetes.container.image", self.config.image)
        conf.set("spark.kubernetes.namespace", self.config.namespace)
        conf.set("spark.kubernetes.authenticate.driver.serviceAccountName", self.config.service_account)
        conf.set("spark.kubernetes.authenticate.serviceAccountName", self.config.service_account)
        conf.set("spark.kubernetes.container.image.pullPolicy", self.config.image_pull_policy)
        
        # Set driver configuration
        conf.set("spark.driver.host", self.config.driver_host)
        conf.set("spark.driver.port", self.config.driver_port)
        conf.set("spark.driver.bindAddress", self.config.driver_bind_address)
        conf.set("spark.driver.blockManager.port", self.config.block_manager_port)
        conf.set("spark.driver.memory", self.config.driver_memory)
        
        # Set executor configuration
        conf.set("spark.executor.instances", str(self.config.executor_instances))
        conf.set("spark.executor.memory", self.config.executor_memory)
        conf.set("spark.executor.cores", str(self.config.executor_cores))
        conf.set("spark.kubernetes.executor.request.cores", str(self.config.executor_cores))
        conf.set("spark.kubernetes.executor.limit.cores", str(self.config.executor_cores))
        
        # Add any extra configuration
        for key, value in self.config.extra_conf.items():
            conf.set(key, value)
        
        logger.info("Created SparkConf with Kubernetes configuration")
        return conf
    
    def create_spark_session(self) -> SparkSession:
        """
        Create a SparkSession with the appropriate configuration.
        
        Returns:
            SparkSession: Configured Spark session
        """
        try:
            conf = self.create_spark_conf()
            spark = SparkSession.builder.config(conf=conf).getOrCreate()
            logger.info("Successfully created SparkSession")
            return spark
        except Exception as e:
            logger.error(f"Failed to create SparkSession: {str(e)}")
            raise
    
    @validate_cluster_size
    def submit(self, job_function: SparkJobFunction[T], *args: Any, **kwargs: Any) -> T:
        """
        Submit a Spark job for execution.
        
        Args:
            job_function: Function that takes a SparkSession and returns a result
            *args: Additional positional arguments to pass to the job function
            **kwargs: Additional keyword arguments to pass to the job function
            
        Returns:
            The result of the job function
        """
        try:
            # Validate configuration before submitting
            self.validate_configuration()
            
            # Create SparkSession if it doesn't exist
            if self.spark is None:
                self.spark = self.create_spark_session()
            
            # Execute the job function
            logger.info(f"Submitting job: {job_function.__name__}")
            result = job_function(self.spark, *args, **kwargs)
            logger.info(f"Job {job_function.__name__} completed successfully")
            
            return result
        except Exception as e:
            logger.error(f"Job submission failed: {str(e)}")
            raise
    
    def __enter__(self):
        """
        Enter the context manager, creating the SparkSession.
        
        Returns:
            KubernetesSparkJob: This instance
        """
        try:
            self.validate_configuration()
            self.spark = self.create_spark_session()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize context: {str(e)}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, stopping the SparkSession.
        
        Args:
            exc_type: Exception type, if an exception was raised
            exc_val: Exception value, if an exception was raised
            exc_tb: Exception traceback, if an exception was raised
        """
        if self.spark is not None:
            try:
                logger.info("Stopping SparkSession")
                self.spark.stop()
                self.spark = None
            except Exception as e:
                logger.warning(f"Error stopping SparkSession: {str(e)}")


# Example usage
def main():
    """Example usage of the KubernetesSparkJob class."""
    # Define a simple job function
    def example_job(spark):
        # Create a simple DataFrame
        data = [("Alice", 34), ("Bob", 45), ("Charlie", 29)]
        df = spark.createDataFrame(data, ["Name", "Age"])
        return df.groupBy().avg("Age").collect()[0][0]
    
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
            avg_age = job.submit(example_job)
            print(f"Average age: {avg_age}")
    except Exception as e:
        print(f"Job failed: {str(e)}")


if __name__ == "__main__":
    main() 