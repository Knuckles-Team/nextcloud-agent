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
  * [](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
    * [Webdav](https://docs.nextcloud.com/server/14/developer_manual/client_apis/WebDAV/index.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html)
      * [Testing requests with curl](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#testing-requests-with-curl)
      * [User metadata](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#user-metadata)
      * [Capabilities API](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#capabilities-api)
      * [Theming capabilities](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#theming-capabilities)
      * [Notifications](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#notifications)
    * [Login Flow](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html) »
  * OCS API’s
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/client_apis/OCS/index.rst)


* * *
# OCS API’s[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#ocs-api-s "Permalink to this headline")
This document provides a quick overview of the OCS API endpoints supported in Nextcloud.
All requests need to provide authentication information, either as a Basic Auth header or by passing a set of valid session cookies, if not stated otherwise.
## Testing requests with curl[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#testing-requests-with-curl "Permalink to this headline")
All OCS requests can be easily tested out using `curl` by specifying the request method (`GET`, `PUT`, etc) and setting a request body where needed.
For example: you can perform a `GET` request to get information about a user:
```
curl -u username:password -X GET 'https://cloud.example.com/ocs/v1.php/...' -H "OCS-APIRequest: true"

```

## User metadata[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#user-metadata "Permalink to this headline")
Since: 11.0.2, 12.0.0
This request returns the available metadata of a user. Admin users can see the information of all users, while a default user only can access it’s own metadata.
```
GET /ocs/v1.php/cloud/users/USERID

```

```
<?xml version="1.0"?>
<ocs>
        <meta>
                <status>ok</status>
                <statuscode>100</statuscode>
                <message>OK</message>
                <totalitems></totalitems>
                <itemsperpage></itemsperpage>
        </meta>
        <data>
                <enabled>true</enabled>
                <quota>
                        <free>338696790016</free>
                        <used>7438874</used>
                        <total>338704228890</total>
                        <relative>0</relative>
                        <quota>-3</quota>
                </quota>
                <email>user@foo.de</email>
                <displayname>admin</displayname>
                <phone></phone>
                <address></address>
                <webpage></webpage>
                <twitter>schiessle</twitter>
        </data>
</ocs>

```

## Capabilities API[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#capabilities-api "Permalink to this headline")
Clients can obtain capabilities provided by the Nextcloud server and its apps via the capabilities OCS API.
```
GET /ocs/v1.php/cloud/capabilities

```

```
<?xml version="1.0"?>
<ocs>
 <meta>
  <status>ok</status>
  <statuscode>100</statuscode>
  <message>OK</message>
  <totalitems></totalitems>
  <itemsperpage></itemsperpage>
 </meta>
 <data>
  <version>
   <major>12</major>
   <minor>0</minor>
   <micro>0</micro>
   <string>12.0 beta 4</string>
   <edition></edition>
  </version>
  <capabilities>
   <core>
    <pollinterval>60</pollinterval>
    <webdav-root>remote.php/webdav</webdav-root>
   </core>
  </capabilities>
 </data>
</ocs>

```

## Theming capabilities[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#theming-capabilities "Permalink to this headline")
Values of the theming app are exposed though the capabilities API, allowing client developers to adjust the look of clients to the theming of different Nextcloud instances.
```
<theming>
    <name>Nextcloud</name>
    <url>https://nextcloud.com</url>
    <slogan>A safe home for all your data</slogan>
    <color>#0082c9</color>
    <logo>http://cloud.example.com/index.php/apps/theming/logo?v=1</logo>
    <background>http://cloud.example.com/index.php/apps/theming/logo?v=1</background>
</theming>

```

The background value can either be an URL to the background image or a hex color value.
## Notifications[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html#notifications "Permalink to this headline")
There is also the [Notifications API](https://github.com/nextcloud/notifications/blob/master/docs/ocs-endpoint-v2.md) As well as documentation on how to [Register a device for push notifications](https://github.com/nextcloud/notifications/blob/5a2d3607952bad675e4057620a9c7de8a7f84f0b/docs/push-v3.md)
[Next ](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html "Login Flow") [](https://docs.nextcloud.com/server/14/developer_manual/client_apis/WebDAV/chunking.html "Chunked file upload")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
