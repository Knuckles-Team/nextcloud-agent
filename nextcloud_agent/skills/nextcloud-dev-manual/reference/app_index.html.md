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
  * [Testing](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html)
  * [App store publishing](https://docs.nextcloud.com/server/14/developer_manual/app/publishing.html)
  * [Code signing](https://docs.nextcloud.com/server/14/developer_manual/app/code_signing.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/index.html)
    * [Intro](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#intro)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#id1)
      * [Requests](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#requests)
      * [View](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#view)
      * [Storage](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#storage)
      * [Authentication & users](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#authentication-users)
      * [Hooks](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#hooks)
      * [Background jobs](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#background-jobs)
      * [Settings](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#settings)
      * [Notifications](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#notifications)
      * [Logging](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#logging)
      * [Migrations](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#migrations)
      * [Repair steps](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#repair-steps)
      * [Testing](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#testing)
      * [PHPDoc class documentation](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#phpdoc-class-documentation)
  * [Design guidelines](https://docs.nextcloud.com/server/14/developer_manual/design/index.html)
  * [Android application development](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html)
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * App development
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/index.rst)


* * *
# App development[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#app-development "Permalink to this headline")
## Intro[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#intro "Permalink to this headline")
Before you start, please check if there already is a similar app in the [App Store](https://apps.nextcloud.com) or the [GitHub organisation](https://github.com/nextcloud/) that you could contribute to. Also, feel free to communicate your idea and plans in the [forum](https://help.nextcloud.com/) so other contributors might join in.
Then, please make sure you have set up a development environment:
  * [Development environment](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html)


Before starting to write an app please read the security and coding guidelines:
  * [Security guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/security.html)
  * [Coding style & general guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/codingguidelines.html)


After this you can start with the tutorial
  * [Tutorial](https://docs.nextcloud.com/server/14/developer_manual/app/tutorial.html)


Once you are ready for publishing, check out the app store process:
  * [App store publishing](https://docs.nextcloud.com/server/14/developer_manual/app/publishing.html)


For enhanced security it is also possible to sign your code:
  * [Code signing](https://docs.nextcloud.com/server/14/developer_manual/app/code_signing.html)


## App development[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#id1 "Permalink to this headline")
Take a look at the changes in this version:
  * [Changelog](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html)


Create a new app:
  * [Create an app](https://docs.nextcloud.com/server/14/developer_manual/app/startapp.html)


Inner parts of an app:
  * [Navigation and pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html)
  * [App metadata](https://docs.nextcloud.com/server/14/developer_manual/app/info.html)
  * [Classloader](https://docs.nextcloud.com/server/14/developer_manual/app/classloader.html)


### Requests[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#requests "Permalink to this headline")
How a request is being processed:
  * [Request lifecycle](https://docs.nextcloud.com/server/14/developer_manual/app/request.html)
  * [Routing](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html)
  * [Middleware](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html)
  * [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html)
  * [Controllers](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html) | [RESTful API](https://docs.nextcloud.com/server/14/developer_manual/app/api.html)


### View[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#view "Permalink to this headline")
The app’s presentation layer:
  * [Templates](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html)
  * [JavaScript](https://docs.nextcloud.com/server/14/developer_manual/app/js.html)
  * [CSS](https://docs.nextcloud.com/server/14/developer_manual/app/css.html)
  * [Translation](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html)
  * [Theming support](https://docs.nextcloud.com/server/14/developer_manual/app/theming.html)


### Storage[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#storage "Permalink to this headline")
Create database tables, run Sql queries, store/retrieve configuration information and access the filesystem:
  * [Database schema](https://docs.nextcloud.com/server/14/developer_manual/app/schema.html)
  * [Database access](https://docs.nextcloud.com/server/14/developer_manual/app/database.html)
  * [Configuration](https://docs.nextcloud.com/server/14/developer_manual/app/configuration.html)
  * [Filesystem](https://docs.nextcloud.com/server/14/developer_manual/app/filesystem.html)


### Authentication & users[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#authentication-users "Permalink to this headline")
Creating, deleting, updating, searching, login and logout:
  * [User management](https://docs.nextcloud.com/server/14/developer_manual/app/users.html)


Writing a two-factor auth provider:
  * [Two-factor providers](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html)


### Hooks[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#hooks "Permalink to this headline")
Listen on events like user creation and execute code:
  * [Hooks](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html)


### Background jobs[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#background-jobs "Permalink to this headline")
Periodically run code in the background:
  * [Background jobs (Cron)](https://docs.nextcloud.com/server/14/developer_manual/app/backgroundjobs.html)


### Settings[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#settings "Permalink to this headline")
An app can register both admin settings as well as personal settings:
  * [Settings](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html)


### Notifications[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#notifications "Permalink to this headline")
An app can send notifications to display to users. It can also retrieve and act upon notifications that are received by the users. See the [documentation of the official Notifications app](https://github.com/nextcloud/notifications#developers).
### Logging[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#logging "Permalink to this headline")
Log to the `data/nextcloud.log`:
  * [Logging](https://docs.nextcloud.com/server/14/developer_manual/app/logging.html)


### Migrations[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#migrations "Permalink to this headline")
Migrations can be used to do database changes which are allowing apps a more granular updating mechanism:
  * [Migrations](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html)


### Repair steps[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#repair-steps "Permalink to this headline")
Repair steps can be used to run code at various stages in app installation, uninstallation, migration and more:
  * [Repair steps](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html)


### Testing[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#testing "Permalink to this headline")
Write automated tests to ensure stability and ease maintenance:
  * [Testing](https://docs.nextcloud.com/server/14/developer_manual/app/testing.html)


### PHPDoc class documentation[¶](https://docs.nextcloud.com/server/14/developer_manual/app/index.html#phpdoc-class-documentation "Permalink to this headline")
Nextcloud class and function documentation:
  * [Nextcloud App API](https://api.owncloud.org/namespaces/OCP.html)


[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html "Changelog") [](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html "Backporting")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
