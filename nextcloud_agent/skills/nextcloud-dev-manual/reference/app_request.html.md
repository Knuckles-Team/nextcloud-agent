[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
  * [Changelog](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html)
  * [Tutorial](https://docs.nextcloud.com/server/14/developer_manual/app/tutorial.html)
  * [Create an app](https://docs.nextcloud.com/server/14/developer_manual/app/startapp.html)
  * [Navigation and pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html)
  * [App metadata](https://docs.nextcloud.com/server/14/developer_manual/app/info.html)
  * [Classloader](https://docs.nextcloud.com/server/14/developer_manual/app/classloader.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/request.html)
    * [Front controller](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#front-controller)
    * [Router](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#router)
    * [Middleware](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#middleware)
    * [Container](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#container)
    * [Controller](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#controller)
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
  * Request lifecycle
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/request.rst)


* * *
# Request lifecycle[¶](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#request-lifecycle "Permalink to this headline")
A typical HTTP request consists of the following:
  * **A URL** : e.g. /index.php/apps/myapp/something
  * **Request Parameters** : e.g. ?something=true&name=tom
  * **A Method** : e.g. GET
  * **Request headers** : e.g. Accept: application/json


The following sections will present an overview over how that request is being processed to provide an in depth view over how Nextcloud works. If you are not interested in the internals or don’t want to execute anything before and after your controller, feel free to skip this section and continue directly with defining [your app’s routes](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html).
## Front controller[¶](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#front-controller "Permalink to this headline")
In the beginning, all requests are sent to Nextcloud’s `index.php` which in turn executes `lib/base.php`. This file inspects the HTTP headers, abstracts away differences between different Web servers and initializes the basic classes. Afterwards the basic apps are being loaded in the following order:
  * Authentication backends
  * Filesystem
  * Logging


The type of the app is determined by inspecting the app’s [configuration file](https://docs.nextcloud.com/server/14/developer_manual/app/info.html) (`appinfo/info.xml`). Loading apps means that the [main file](https://docs.nextcloud.com/server/14/developer_manual/app/init.html) (`appinfo/app.php`) of each installed app is being loaded and executed. That means that if you want to execute code before a specific app is being run, you can place code in your app’s [Navigation and pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html) file.
Afterwards the following steps are performed:
  * Try to authenticate the user
  * Load and execute all the remaining apps’ [Navigation and pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html) files
  * Load and run all the routes in the apps’ `appinfo/routes.php`
  * Execute the router


## Router[¶](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#router "Permalink to this headline")
The router parses the [app’s routing files](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html) (`appinfo/routes.php`), inspects the request’s **method** and **url** , queries the controller from the [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html) and then passes control to the dispatcher. The dispatcher is responsible for running the hooks (called Middleware) before and after the controller, executing the controller method and rendering the output.
## Middleware[¶](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#middleware "Permalink to this headline")
A [Middleware](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html) is a convenient way to execute common tasks such as custom authentication before or after a [controller method](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html) is being run. You can execute code at the following locations:
  * before the call of the controller method
  * after the call of the controller method
  * after an exception is thrown (also if it is thrown from a middleware, e.g. if an authentication fails)
  * before the output is rendered


## Container[¶](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#container "Permalink to this headline")
The [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html) is the place where you define all of your classes and in particular all of your controllers. The container is responsible for assembling all of your objects (instantiating your classes) that should only have one single instance without relying on globals or singletons. If you want to know more about why you should use it and what the benefits are, read up on the topic in [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html).
## Controller[¶](https://docs.nextcloud.com/server/14/developer_manual/app/request.html#controller "Permalink to this headline")
The [controller](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html) contains the code that you actually want to run after a request has come in. Think of it like a callback that is executed if everything before went fine.
The controller returns a response which is then run through the middleware again (afterController and beforeOutput hooks are being run), HTTP headers are being set and the response’s render method is being called and printed.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html "Routing") [](https://docs.nextcloud.com/server/14/developer_manual/app/classloader.html "Classloader")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
