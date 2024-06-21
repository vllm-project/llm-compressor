import logging
import runpy
import unittest
from typing import Optional

from tests.data import CustomTestConfig

_LOGGER = logging.getLogger(__name__)


class CustomTestCase(unittest.TestCase):
    """
    CustomTestCase class. All custom test classes written should inherit from this
    class. They will be subsequently tested in the test_custom_class function defined
    within the CustomIntegrationTest.
    """

    ...


# TODO: consider breaking this up into two classes, similar to non-custom
# integration tests. Could then make use of parameterize_class instead
class CustomIntegrationTest(unittest.TestCase):
    """
    Base Class for all custom integration tests.
    """

    custom_scripts_directory: str = None
    custom_class_directory: str = None

    def test_custom_scripts(self, config: Optional[CustomTestConfig] = None):
        """
        This test case will run all custom python scripts that reside in the directory
        defined by custom_scripts_directory. For each custom python script, there
        should be a corresponding yaml file which consists of the values defined by
        the dataclass CustomTestConfig, including the field script_path which is
        populated with the name of the python script. The test will fail if any
        of the defined assertions in the script fail

        :param config: config defined by the CustomTestConfig dataclass

        """
        if config is None:
            self.skipTest("No custom scripts found. Testing test")
        script_path = f"{self.custom_scripts_directory}/{config.script_path}"
        runpy.run_path(script_path)

    def test_custom_class(self, config: Optional[CustomTestConfig] = None):
        """
        This test case will run all custom test classes that reside in the directory
        defined by custom_class_directory. For each custom test class, there
        should be a corresponding yaml file which consists of the values defined by
        the dataclass CustomTestConfig, including the field script_path which is
        populated with the name of the python script. The test will fail if any
        of the defined tests in the custom class fail.

        :param config: config defined by the CustomTestConfig dataclass

        """
        if config is None:
            self.skipTest("No custom class found. Testing test")
        loader = unittest.TestLoader()
        tests = loader.discover(self.custom_class_directory, pattern=config.script_path)
        testRunner = unittest.runner.TextTestRunner()
        output = testRunner.run(tests)
        for out in output.errors:
            raise Exception(output[-1])

        for out in output.failures:
            _LOGGER.error(out[-1])
            assert False
