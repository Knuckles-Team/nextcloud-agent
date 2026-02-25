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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/js.html)
    * [Sending the CSRF token](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#sending-the-csrf-token)
    * [Generating URLs](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#generating-urls)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#extending-core-parts)
      * [Extending the “new” menu in the files app](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#extending-the-new-menu-in-the-files-app)
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
  * JavaScript
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/js.rst)


* * *
# JavaScript[¶](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#javascript "Permalink to this headline")
The JavaScript files reside in the **js/** folder and should be included in the template:
```
<?php
// add one file
script('myapp', 'script');  // adds js/script.js

// add multiple files in the same app
script('myapp', array('script', 'navigation'));  //  adds js/script.js js/navigation.js

// add vendor files (also allows the array syntax)
vendor_script('myapp', 'script');  //  adds vendor/script.js

```

If the script file is only needed when the file list is displayed, you should listen to the `OCA\Files::loadAdditionalScripts` event:
```
<?php
$eventDispatcher = \OC::$server->getEventDispatcher();
$eventDispatcher->addListener('OCA\Files::loadAdditionalScripts', function() {
  script('myapp', 'script');  // adds js/script.js
  vendor_script('myapp', 'script');  //  adds vendor/script.js
});

```

## Sending the CSRF token[¶](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#sending-the-csrf-token "Permalink to this headline")
If any other JavaScript request library than jQuery is being used, the requests need to send the CSRF token as an HTTP header named **requesttoken**. The token is available in the global variable **oc_requesttoken**.
For AngularJS the following lines would need to be added:
```
var app = angular.module('MyApp', []).config(['$httpProvider', function($httpProvider) {
    $httpProvider.defaults.headers.common.requesttoken = oc_requesttoken;
}]);

```

## Generating URLs[¶](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#generating-urls "Permalink to this headline")
To send requests to Nextcloud the base URL where Nextcloud is currently running is needed. To get the base URL use:
```
var baseUrl = OC.generateUrl('');

```

Full URLs can be generated by using:
```
var authorUrl = OC.generateUrl('/apps/myapp/authors/1');

```

## Extending core parts[¶](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#extending-core-parts "Permalink to this headline")
It is possible to extend components of the core web UI. The following examples should show how this is possible.
### Extending the “new” menu in the files app[¶](https://docs.nextcloud.com/server/14/developer_manual/app/js.html#extending-the-new-menu-in-the-files-app "Permalink to this headline")
New in version 9.0.
```
var myFileMenuPlugin = {
    attach: function (menu) {
        menu.addMenuEntry({
            id: 'abc',
            displayName: 'Menu display name',
            templateName: 'templateName.ext',
            iconClass: 'icon-filetype-text',
            fileType: 'file',
            actionHandler: function () {
                console.log('do something here');
            }
        });
    }
};
OC.Plugins.register('OCA.Files.NewFileMenu', myFileMenuPlugin);

```

This will register a new menu entry in the “New” menu of the files app. The method `attach()` is called once the menu is built. This usually happens right after the click on the button.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/css.html "CSS") [](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html "Templates")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
