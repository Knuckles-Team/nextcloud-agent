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
  * [App development](https://docs.nextcloud.com/server/14/developer_manual/app/index.html)
  * [Design guidelines](https://docs.nextcloud.com/server/14/developer_manual/design/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html)
    * [Android Nextcloud client development](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html#android-nextcloud-client-development)
    * [](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html#nextcloud-android-library)
      * [Library installation](https://docs.nextcloud.com/server/14/developer_manual/android_library/library_installation.html)
      * [Examples](https://docs.nextcloud.com/server/14/developer_manual/android_library/examples.html)
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * Android application development
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/android_library/index.rst)


* * *
# Android application development[¶](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html#android-application-development "Permalink to this headline")
Nextcloud provides an official Nextcloud Android client, which gives its users access to their files on their Nextcloud. It also includes functionality like automatically uploading pictures and videos to Nextcloud.
For third party application developers, Nextcloud offers the Nextcloud Android library under the MIT license.
## Android Nextcloud client development[¶](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html#android-nextcloud-client-development "Permalink to this headline")
If you are interested in working on the Nextcloud Android client, you can find the source code [in GitHub](https://github.com/nextcloud/android/). The setup and process of contribution is [documented here](https://github.com/nextcloud/android/blob/master/SETUP.md).
You might want to start with doing one or two [good first issues](https://github.com/nextcloud/android/labels/good%20first%20issue) to get into the code and note our [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html).
## Nextcloud Android library[¶](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html#nextcloud-android-library "Permalink to this headline")
This document will describe how to the use Nextcloud Android Library. The Nextcloud Android Library allows a developer to communicate with any Nextcloud server; among the features included are file synchronization, upload and download of files, delete or rename files and folders, etc.
This library may be added to a project and seamlessly integrates any application with Nextcloud.
The tool needed is any IDE for Android; the preferred IDE at the moment is Android Studio.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/android_library/library_installation.html "Library installation") [](https://docs.nextcloud.com/server/14/developer_manual/design/icons.html "Icons")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
