[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
    * [Community code of conduct](https://docs.nextcloud.com/server/14/developer_manual/general/code-of-conduct.html)
    * [Development environment](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html)
    * [Security guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/security.html)
    * [Coding style & general guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/codingguidelines.html)
    * [Performance considerations](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html)
    * [Debugging](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html)
      * [General](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html#general)
      * [Process](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html#process)
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
  * [Android application development](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html)
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html) »
  * Backporting
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/general/backporting.rst)


* * *
# Backporting[¶](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html#backporting "Permalink to this headline")
## General[¶](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html#general "Permalink to this headline")
We backport important fixes and improvements from the current master release to get them to our users faster.
## Process[¶](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html#process "Permalink to this headline")
We mostly consider bug fixes for back porting. Occasionally, important changes to the API can be backported to make it easier for developers to keep their apps working between major releases. If you think a pull request (PR) is relevant for the stable release, go through these steps:
  1. Make sure the PR is merged to master
  2. Ask Frank (**@karlitschek**), if the code should be backported and add the label [backport-request](https://github.com/nextcloud/server/labels/backport-request) to the PR
  3. If Frank approves, create a new branch based on the respective stable branch (stable10 for the 10.0.x series), cherry-pick the needed commits to that branch and create a PR on GitHub.
  4. Specify the corresponding milestone for that series (10.0.x-next-maintenance for the 10.0.x series) to this PR and reference the original PR in there. This enables the QA team to find the backported items for testing and having the original PR with detailed description linked.


Note
Before each patch release there is a freeze to be able to test everything as a whole without pulling in new changes. While this freeze is active a backport isn’t allowed and has to wait for the next patch release.
The QA team will try to reproduce all the issues with the X.Y.Z-next-maintenance milestone on the relevant release and verify it is fixed by the patch release (and doesn’t cause new problems). Once the patch release is out, the post-fix -next-maintenance is removed and a new -next-maintenance milestone is created for that series.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/index.html "App development") [](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html "Debugging")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
