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
    * [OCS API’s](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html)
      * [Opening the webview](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#opening-the-webview)
      * [Login in the user](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#login-in-the-user)
      * [Obtaining the login credentials](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#obtaining-the-login-credentials)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html) »
  * Login Flow
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/client_apis/LoginFlow/index.rst)


* * *
# Login Flow[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#login-flow "Permalink to this headline")
This document provides a quick overview of the new login flow that should be used by clients to obtain login credentials. This will assure that each client gets it own set of credentials. This has several advantages:
  1. The client never stores the password of the user
  2. The user can revoke on a per client basis from the web


## Opening the webview[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#opening-the-webview "Permalink to this headline")
The client should open a webview to `<server>/index.php/login/flow`. Be sure to set the `OCS-APIREQUEST` header to `true`.
The client will register an URL handler to catch urls of the `nc` protocol. This is required to obtain the credentials in the final stage.
This should be a one time webview. Which means:
  * There should be no cookies set when creating the webview
  * Passwords should not be stored
  * No state should be preserved after the webview has terminated


To have a good user experince please consider the following things:
  * set a proper `ACCEPT_LANGUAGE` header
  * set a proper `USER_AGENT` header


## Login in the user[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#login-in-the-user "Permalink to this headline")
The user will now see a webpage telling them they will grant access to `USER_AGENT`. When they follow the steps they will be asked to login. If they have two factor authentication enabled they will require this to login. But since this is all in the webview itself the client does not need to care about this.
## Obtaining the login credentials[¶](https://docs.nextcloud.com/server/14/developer_manual/client_apis/LoginFlow/index.html#obtaining-the-login-credentials "Permalink to this headline")
On the final login the server will do a redirect to a url of the following format:
```
nc://login/server:<server>&user:<loginname>&password:<password>

```

  * server: The address of the server to connect to. The server may specify a protocol (http or https). If no protocol is specified the client will assume https.
  * loginname: The username that the client must use to login. **Note:** Keep in mind that this is the loginname and could be different from the username. For example the email address could be used to login but not for generating the webdav URL. You could fetch the actual username from the OCS API endpoint `<server>/ocs/v1.php/cloud/user`.
  * password: The password that the client must use to login and store securely


This information will be used by the client to create a new account. After this the webview is destroyed including all the state the webview holds.
Note
On Nextcloud 12 the returned server is just the server address without any possible subfolder. This is corrected in Nextcloud 13.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/core/index.html "Core development") [](https://docs.nextcloud.com/server/14/developer_manual/client_apis/OCS/index.html "OCS API’s")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
