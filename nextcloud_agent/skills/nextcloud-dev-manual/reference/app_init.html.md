[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
  * [Changelog](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html)
  * [Tutorial](https://docs.nextcloud.com/server/14/developer_manual/app/tutorial.html)
  * [Create an app](https://docs.nextcloud.com/server/14/developer_manual/app/startapp.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/init.html)
    * [Adding a navigation entry](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#adding-a-navigation-entry)
    * [Further pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#further-pre-app-configuration)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#best-practice)
      * [appinfo/app.php](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#appinfo-app-php)
      * [lib/AppInfo/Application.php](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#lib-appinfo-application-php)
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
  * Navigation and pre-app configuration
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/init.rst)


* * *
# Navigation and pre-app configuration[¶](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#navigation-and-pre-app-configuration "Permalink to this headline")
## Adding a navigation entry[¶](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#adding-a-navigation-entry "Permalink to this headline")
Navigation entries for apps can be created by adding a navigation section to the `appinfo/info.xml` file, containing the name, order and route the navigation entry should link to. For details on the XML schema check the [app store documentation](https://nextcloudappstore.readthedocs.io/en/latest/developer.html#info-xml).
```
<navigation>
    <name>MyApp</name>
    <route>myapp.page.index</route>
    <order>0</order>
</navigation>

```

## Further pre-app configuration[¶](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#further-pre-app-configuration "Permalink to this headline")
The `appinfo/app.php` is the first file that is loaded and executed in Nextcloud. Depending on the purpose of the app it is usually used to setup things that need to be available on every request to the server, like [Background jobs (Cron)](https://docs.nextcloud.com/server/14/developer_manual/app/backgroundjobs.html) and [Hooks](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html) registrations. This is how an example `appinfo/app.php` could look like:
```
<?php

// execute OCA\MyApp\BackgroundJob\Task::run when cron is called
\OC::$server->getJobList()->add('OCA\MyApp\BackgroundJob\Task');

// execute OCA\MyApp\Hooks\User::deleteUser before a user is being deleted
\OCP\Util::connectHook('OC_User', 'pre_deleteUser', 'OCA\MyApp\Hooks\User', 'deleteUser');

```

Although it is also possible to include [JavaScript](https://docs.nextcloud.com/server/14/developer_manual/app/js.html) or [CSS](https://docs.nextcloud.com/server/14/developer_manual/app/css.html) for other apps by placing the **addScript** or **addStyle** functions inside this file, it is strongly discouraged, because the file is loaded on each request (also such requests that do not return HTML, but e.g. json or webdav).
```
<?php

\OCP\Util::addScript('myapp', 'script');  // include js/script.js for every app
\OCP\Util::addStyle('myapp', 'style');  // include css/style.css for every app

```

## Best practice[¶](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#best-practice "Permalink to this headline")
A common way to have a cleaner code structure is to create a class Application in `lib/AppInfo/Application.php` that will then execute your setup of hooks or background tasks. You can then just call it in your `appinfo/app.php`. That way you can also make use of Nextclouds dependency injection feature and properly test those methods.
### appinfo/app.php[¶](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#appinfo-app-php "Permalink to this headline")
```
<?php

$app = new \OCA\MyApp\AppInfo\Application();
$app->registerHooks();

```

### lib/AppInfo/Application.php[¶](https://docs.nextcloud.com/server/14/developer_manual/app/init.html#lib-appinfo-application-php "Permalink to this headline")
```
<?php
namespace OCA\MyApp\AppInfo;

use OCP\AppFramework\App;

class Application extends App {

    public function registerHooks() {
        \OCP\Util::connectHook('OC_User', 'pre_deleteUser', 'OCA\MyApp\Hooks\User', 'deleteUser');
    }

}

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/info.html "App metadata") [](https://docs.nextcloud.com/server/14/developer_manual/app/startapp.html "Create an app")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
