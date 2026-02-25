[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#deprecations)
      * [2018](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id1)
      * [2017](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id2)
      * [2016](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id3)
      * [2015](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id4)
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
  * [App development](https://docs.nextcloud.com/server/14/developer_manual/app/index.html) »
  * Changelog
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/changelog.rst)


* * *
# Changelog[¶](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#changelog "Permalink to this headline")
## Deprecations[¶](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#deprecations "Permalink to this headline")
This is a deprecation roadmap which lists all current deprecation targets and will be updated from release to release. This lists the year when a specific method or class will be removed.
Note
Deprecations on interfaces also affect the implementing classes!
### 2018[¶](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id1 "Permalink to this headline")
  * **OCP\App::setActiveNavigationEntry** has been deprecated in favour of **\OCP\INavigationManager**
  * **OCP\BackgroundJob::registerJob** has been deprecated in favour of **OCP\BackgroundJob\IJobList**
  * **OCP\Contacts** functions has been deprecated in favour of **\OCP\Contacts\IManager**
  * **OCP\DB** functions have been deprecated in favour of the ones in **\OCP\IDBConnection**
  * **OCP\Files::tmpFile** has been deprecated in favour of **\OCP\ITempManager::getTemporaryFile**
  * **OCP\Files::tmpFolder** has been deprecated in favour of **\OCP\ITempManager::getTemporaryFolder**
  * **\OCP\IServerContainer::getDb** has been deprecated in favour of **\OCP\IServerContainer::getDatabaseConnection**
  * **\OCP\IServerContainer::getHTTPHelper** has been deprecated in favour of **\OCP\Http\Client\IClientService**
  * Legacy applications not using the AppFramework are now likely to use the deprecated **OCP\JSON** and **OCP\Response** code:
    * **\OCP\JSON** has been completely deprecated in favour of the AppFramework. Developers shall use the AppFramework instead of using the legacy **OCP\JSON** code. This allows testable controllers and is highly encouraged.
    * **\OCP\Response** has been completely deprecated in favour of the AppFramework. Developers shall use the AppFramework instead of using the legacy **OCP\JSON** code. This allows testable controllers and is highly encouraged.
  * Diverse **OCP\Users** function got deprecated in favour of **OCP\IUserManager** :
    * **OCP\Users::getUsers** has been deprecated in favour of **OCP\IUserManager::search**
    * **OCP\Users::getDisplayName** has been deprecated in favour of **OCP\IUserManager::getDisplayName**
    * **OCP\Users::getDisplayNames** has been deprecated in favour of **OCP\IUserManager::searchDisplayName**
    * **OCP\Users::userExists** has been deprecated in favour of **OCP\IUserManager::userExists**
  * Various static **OCP\Util** functions have been deprecated:
    * **OCP\Util::linkToRoute** has been deprecated in favour of **\OCP\IURLGenerator::linkToRoute**
    * **OCP\Util::linkTo** has been deprecated in favour of **\OCP\IURLGenerator::linkTo**
    * **OCP\Util::imagePath** has been deprecated in favour of **\OCP\IURLGenerator::imagePath**
    * **OCP\Util::isValidPath** has been deprecated in favour of **\OCP\IURLGenerator::imagePath**
  * [OCP\AppFramework\IAppContainer](https://github.com/nextcloud/server/blob/stable9/lib/public/appframework/iappcontainer.php): methods **getCoreApi** and **log**
  * [OCP\AppFramework\IApi](https://github.com/nextcloud/server/blob/stable9/lib/public/appframework/iapi.php): full class


### 2017[¶](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id2 "Permalink to this headline")
  * **OCP\IDb** : This interface and the implementing classes will be removed in favor of **OCP\IDbConnection**. Various layers in between have also been removed to be consistent with the PDO classes. This leads to the following changes:


>   * Replace all calls on the db using **getInsertId** with **lastInsertId**
>   * Replace all calls on the db using **prepareQuery** with **prepare**
>   * The **__construct** method of **OCP\AppFramework\Db\Mapper** no longer requires an instance of **OCP\IDb** but an instance of **OCP\IDbConnection**
>   * The **execute** method on **OCP\AppFramework\Db\Mapper** no longer returns an instance of **OC_DB_StatementWrapper** but an instance of **PDOStatement**
>

### 2016[¶](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id3 "Permalink to this headline")
  * The following methods have been moved into the **OCP\Template:: <method>** class instead of being namespaced directly:


>   * **OCP\image_path**
>   * **OCP\mimetype_icon**
>   * **OCP\preview_icon**
>   * **OCP\publicPreview_icon**
>   * **OCP\human_file_size**
>   * **OCP\relative_modified_date**
>   * **OCP\html_select_options**
>

  * **OCP\simple_file_size** has been deprecated in favour of **OCP\Template::human_file_size**
  * The **OCP\PERMISSION_ <permission>** and **OCP\FILENAME_INVALID_CHARS** have been moved to **OCP\Constants:: <old name>**
  * The **OC_GROUP_BACKEND_ <method>** and **OC_USER_BACKEND_ <method>** have been moved to **OC_Group_Backend:: <method>** and **OC_User_Backend:: <method>** respectively
  * [OCP\AppFramework\Controller](https://github.com/nextcloud/server/blob/stable9/lib/public/appframework/controller.php): methods **params** , **getParams** , **method** , **getUploadedFile** , **env** , **cookie** , **render**


### 2015[¶](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html#id4 "Permalink to this headline")
  * [\OC\Preferences](https://github.com/nextcloud/server/commit/909a53e087b7815ba9cd814eb6c22845ef5b48c7) and [\OC_Preferences](https://github.com/nextcloud/server/commit/4df7c0a1ed52ed1922116686cb5ad8da2544c997)


[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/tutorial.html "Tutorial") [](https://docs.nextcloud.com/server/14/developer_manual/app/index.html "App development")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
