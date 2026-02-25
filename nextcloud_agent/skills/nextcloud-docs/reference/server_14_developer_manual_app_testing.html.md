[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
  * [Changelog](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html)
  * [Tutorial](https://docs.nextcloud.com/server/14/developer_manual/app/tutorial.html)
  * [Create an app](https://docs.nextcloud.com/server/14/developer_manual/app/startapp.html)
  * [Navigation and pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html)
  * [App metadata](https://docs.nextcloud.com/server/14/developer_manual/app/info.html)
  * [Classloader](https://docs.nextcloud.com/server/14/developer_manual/app/classloader.html)
  * [Request lifecycle](https://docs.nextcloud.com/server/14/developer_manual/app/request.html)
  * [Routing](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html)
  * [Middleware](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html)
  * [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html)
  * [Controllers](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html)
  * [RESTful API](https://docs.nextcloud.com/server/14/developer_manual/app/api.html)
  * [Templates](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html)
  * [JavaScript](https://docs.nextcloud.com/server/14/developer_manual/app/js.html)
  * [CSS](https://docs.nextcloud.com/server/14/developer_manual/app/css.html)
  * [Translation](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html)
  * [Theming support](https://docs.nextcloud.com/server/14/developer_manual/app/theming.html)
  * [Database schema](https://docs.nextcloud.com/server/14/developer_manual/app/schema.html)
  * [Database access](https://docs.nextcloud.com/server/14/developer_manual/app/database.html)
  * [Configuration](https://docs.nextcloud.com/server/14/developer_manual/app/configuration.html)
  * [Filesystem](https://docs.nextcloud.com/server/14/developer_manual/app/filesystem.html)
  * [AppData](https://docs.nextcloud.com/server/14/developer_manual/app/appdata.html)
  * [User management](https://docs.nextcloud.com/server/14/developer_manual/app/users.html)
  * [Two-factor providers](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html)
  * [Hooks](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html)
  * [Background jobs (Cron)](https://docs.nextcloud.com/server/14/developer_manual/app/backgroundjobs.html)
  * [Settings](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html)
  * [Logging](https://docs.nextcloud.com/server/14/developer_manual/app/logging.html)
  * [Migrations](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html)
  * [Repair steps](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html)
    * [PHP](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html#php)
  * [App store publishing](https://docs.nextcloud.com/server/14/developer_manual/app/publishing.html)
  * [Code signing](https://docs.nextcloud.com/server/14/developer_manual/app/code_signing.html)
  * [App development](https://docs.nextcloud.com/server/14/developer_manual/app/index.html)
  * [Design guidelines](https://docs.nextcloud.com/server/14/developer_manual/design/index.html)
  * [Android application development](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html)
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [App development](https://docs.nextcloud.com/server/14/developer_manual/app/index.html) »
  * Testing
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/testing.rst)


* * *
# Testing[¶](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html#testing "Permalink to this headline")
All PHP classes can be tested with [PHPUnit](http://phpunit.de/), JavaScript can be tested by using [Karma](http://karma-runner.github.io).
## PHP[¶](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html#php "Permalink to this headline")
The PHP tests go into the **tests/** directory and PHPUnit can be run with:
```
phpunit tests/

```

When writing your own tests, please ensure that PHPUnit bootstraps from `tests/bootstrap.php`, to set up various environment variables and autoloader registration correctly. Without this, you will see errors as the Nextcloud autoloader security policy prevents access to the tests/ subdirectory. This can be configured in your `phpunit.xml` file as follows:
```
<phpunit bootstrap="../../tests/bootstrap.php">

```

PHP classes should be tested by accessing them from the container to ensure that the container is wired up properly. Services that should be mocked can be replaced directly in the container.
A test for the **AuthorStorage** class in [Filesystem](https://docs.nextcloud.com/server/14/developer_manual/app/filesystem.html):
```
<?php
namespace OCA\MyApp\Storage;

class AuthorStorage {

    private $storage;

    public function __construct($storage){
        $this->storage = $storage;
    }

    public function getContent($id) {
        // check if file exists and write to it if possible
        try {
            $file = $this->storage->getById($id);
            if($file instanceof \OCP\Files\File) {
                return $file->getContent();
            } else {
                throw new StorageException('Can not read from folder');
            }
        } catch(\OCP\Files\NotFoundException $e) {
            throw new StorageException('File does not exist');
        }
    }
}

```

would look like this:
```
<?php
// tests/Storage/AuthorStorageTest.php
namespace OCA\MyApp\Tests\Storage;

class AuthorStorageTest extends \Test\TestCase {

    private $container;
    private $storage;

    protected function setUp() {
        parent::setUp();

        $app = new \OCA\MyApp\AppInfo\Application();
        $this->container = $app->getContainer();
        $this->storage = $storage = $this->getMockBuilder('\OCP\Files\Folder')
            ->disableOriginalConstructor()
            ->getMock();

        $this->container->registerService('RootStorage', function($c) use ($storage) {
            return $storage;
        });
    }

    /**
     * @expectedException \OCA\MyApp\Storage\StorageException
     */
    public function testFileNotFound() {
        $this->storage->expects($this->once())
            ->method('get')
            ->with($this->equalTo(3))
            ->will($this->throwException(new \OCP\Files\NotFoundException()));

        $this->container['AuthorStorage']->getContent(3);
    }

}

```

Make sure to extend the `\Test\TestCase` class with your test and always call the parent methods, when overwriting `setUp()`, `setUpBeforeClass()`, `tearDown()` or `tearDownAfterClass()` method from the TestCase. These methods set up important stuff and clean up the system after the test, so the next test can run without side effects, like remaining files and entries in the file cache, etc.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/publishing.html "App store publishing") [](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html "Repair steps")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
