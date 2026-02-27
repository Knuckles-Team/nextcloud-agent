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
  * [Android application development](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html)
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
    * [Translation](https://docs.nextcloud.com/server/14/developer_manual/core/translation.html)
    * [Unit-Testing](https://docs.nextcloud.com/server/14/developer_manual/core/unit-testing.html)
    * [Theming Nextcloud](https://docs.nextcloud.com/server/14/developer_manual/core/theming.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html)
      * [Introduction](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#introduction)
      * [](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#usage)
        * [Registering methods](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#registering-methods)
        * [Returning data](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#returning-data)
        * [Authentication & basics](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#authentication-basics)
        * [Output](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#output)
        * [Statuscodes](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#statuscodes)
    * [OCS Share API](https://docs.nextcloud.com/server/14/developer_manual/core/ocs-share-api.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html) »
  * External API
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/core/externalapi.rst)


* * *
# External API[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#external-api "Permalink to this headline")
## Introduction[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#introduction "Permalink to this headline")
The external API inside Nextcloud allows third party developers to access data provided by Nextcloud apps. Nextcloud follows the [OpenCloudMesh specification](https://lukasreschke.github.io/OpenCloudMeshSpecification/) (draft).
## Usage[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#usage "Permalink to this headline")
### Registering methods[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#registering-methods "Permalink to this headline")
Methods are registered inside the `appinfo/routes.php` by returning an array holding the endpoint meta data.
```
<?php

'ocs' => [
    // Apps
    ['name' => 'Bar#getFoo', 'url' => '/foobar', 'verb' => 'GET'],
];

```

### Returning data[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#returning-data "Permalink to this headline")
Once the API backend has matched your URL, your callable function as defined in **BarController::getFoo** will be executed. The AppFramework will make sure that send parameters are provided to the method based on its declaration. To return data back to the client, you should return an instance of (a subclass of) `OCPAppFrameworkHttpResponse`, typically `OCSResponse`. The API backend will then use this to construct the XML or JSON response.
### Authentication & basics[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#authentication-basics "Permalink to this headline")
Because REST is stateless you have to send user and password each time you access the API. Therefore running Nextcloud **with SSL is highly recommended** ; otherwise **everyone in your network can log your credentials** :
```
https://user:password@example.com/ocs/v1.php/apps/yourapp

```

### Output[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#output "Permalink to this headline")
The output defaults to XML. If you want to get JSON append this to the URL:
```
?format=json

```

Or set the proper Accept header:
```
Accept: application/json

```

Output from the application is wrapped inside a **data** element:
**XML** :
```
<?xml version="1.0"?>
<ocs>
 <meta>
  <status>ok</status>
  <statuscode>100</statuscode>
  <message/>
 </meta>
 <data>
   <!-- data here -->
 </data>
</ocs>

```

**JSON** :
```
{
  "ocs": {
    "meta": {
      "status": "ok",
      "statuscode": 100,
      "message": null
    },
    "data": {
      // data here
    }
  }
}

```

### Statuscodes[¶](https://docs.nextcloud.com/server/14/developer_manual/core/externalapi.html#statuscodes "Permalink to this headline")
The statuscode can be any of the following numbers:
  * **100** - successful
  * **996** - server error
  * **997** - not authorized
  * **998** - not found
  * **999** - unknown error


[Next ](https://docs.nextcloud.com/server/14/developer_manual/core/ocs-share-api.html "OCS Share API") [](https://docs.nextcloud.com/server/14/developer_manual/core/theming.html "Theming Nextcloud")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
