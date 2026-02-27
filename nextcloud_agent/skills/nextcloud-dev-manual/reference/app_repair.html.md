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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#creating-a-repair-step)
      * [Outputting information](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#outputting-information)
    * [Register a repair-step](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#register-a-repair-step)
    * [Repair-step types](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#repair-step-types)
  * [Testing](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html)
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
  * Repair steps
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/repair.rst)


* * *
# Repair steps[¶](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#repair-steps "Permalink to this headline")
Repair steps are methods which are executed by Nextcloud on certain events which directly affect the app. You can use these repair steps to run code when your app is installed, uninstalled, upgraded etc. It’s called repair steps because they are frequently used to fix things automatically.
Note
Don’t use the `install.php`, `update.php` and `preupdate.php` files anymore! This method is deprecated and is known to cause issues.
## Creating a repair step[¶](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#creating-a-repair-step "Permalink to this headline")
A repair step is an implementation of the `OCP\Migration\IRepairStep` interface. By convention these classes are placed in the **lib/Migration** directory. The following repairstep will log a message when executed.
```
<?php
namespace OCA\MyApp\Migration;

use OCP\Migration\IOutput;
use OCP\Migration\IRepairStep;
use OCP\ILogger;

class MyRepairStep implements IRepairStep {

      /** @var ILogger */
      protected $logger;

      public function __construct(ILogger $logger) {
              $this->logger = $logger;
      }

      /**
       * Returns the step's name
       */
      public function getName() {
              return 'A demonstration repair step!';
      }

      /**
       * @param IOutput $output
       */
      public function run(IOutput $output) {
              $this->logger->warning("Hello world from MyRepairStep!", ["app" => "MyApp"]);
      }
}

```

### Outputting information[¶](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#outputting-information "Permalink to this headline")
A repair step can generate information while running, using the `OCP\Migration\IOutput` parameter to the `run` method. Using the `info` and `warning` methods a message can be shown in the console. In order to show a progressbar, firstly call the `startProgress` method. The maximum number of steps can be adjusted by passing it as argument to the `startProgress` method. After every step run the `advance` method. Once all steps are finished run the `finishProgress` method.
The following function will sleep for 10 seconds and show the progress:
```
<?php
/**
 * @param IOutput $output
 */
public function run(IOutput $output) {
      $output->info("This step will take 10 seconds.");
      $output->startProgress(10);
      for ($i = 0; $i < 10; $i++) {
              sleep(1);
              $output->advance(1);
      }
      $output->finishProgress();
}

```

## Register a repair-step[¶](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#register-a-repair-step "Permalink to this headline")
To register a repair-step in Nextcloud you have to define the repair-setp in the `info.xml` file. The following example registers a repair-step which will be executed after installation of the app:
```
<?xml version="1.0"?>
<info xmlns:xsi= "http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="https://apps.nextcloud.com/schema/apps/info.xsd">
      <id>myapp</id>
      <name>My App</name>
      <summary>A test app</summary>
      ...
      <repair-steps>
              <install>
                      <step>OCA\MyApp\Migration\MyRepairStep</step>
              </install>
      </repair-steps>
</info>

```

## Repair-step types[¶](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html#repair-step-types "Permalink to this headline")
The following repair steps are available:
  * `install` This repair step will be executed when installing the app. This means it is executed every time the app is enabled (using the Web interface or the CLI).
  * `uninstall` This repair step will be executed when uninstalling the app, and when disabling the app.
  * `pre-migration` This repair step will be executed just before the database is migrated during an update of the app.
  * `post-migration` This repair step will be executed just after the database is migrated during an update of the app. This repair step will also be executed when running the `occ maintenance:repair` command
  * `live-migration` This repair step will be scheduled to be run in the background (e.g. using cron), therefore it is unpredictable when it will run. If the job isn’t required right after the update of the app and the job would take a long time this is the best choice.


[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html "Testing") [](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html "Migrations")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
