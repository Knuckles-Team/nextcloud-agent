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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/api.html)
    * [Modifying the CORS headers](https://docs.nextcloud.com/server/14/developer_manual/app/api.html#modifying-the-cors-headers)
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
  * RESTful API
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/api.rst)


* * *
# RESTful API[¶](https://docs.nextcloud.com/server/14/developer_manual/app/api.html#restful-api "Permalink to this headline")
Offering a RESTful API is not different from creating a [route](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html) and [controllers](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html) for the web interface. It is recommended though to inherit from ApiController and add **@CORS** annotations to the methods so that [web applications will also be able to access the API](https://developer.mozilla.org/en-US/docs/Web/HTTP/Access_control_CORS).
```
<?php
namespace OCA\MyApp\Controller;

use \OCP\AppFramework\ApiController;
use \OCP\IRequest;

class AuthorApiController extends ApiController {

    public function __construct($appName, IRequest $request) {
        parent::__construct($appName, $request);
    }

    /**
     * @CORS
     */
    public function index() {

    }

}

```

CORS also needs a separate URL for the preflighted **OPTIONS** request that can easily be added by adding the following route:
```
<?php
// appinfo/routes.php
array(
    'name' => 'author_api#preflighted_cors',
    'url' => '/api/1.0/{path}',
    'verb' => 'OPTIONS',
    'requirements' => array('path' => '.+')
)

```

Keep in mind that multiple apps will likely depend on the API interface once it is published and they will move at different speeds to react to changes implemented in the API. Therefore it is recommended to version the API in the URL to not break existing apps when backwards incompatible changes are introduced:
```
/index.php/apps/myapp/api/1.0/resource

```

## Modifying the CORS headers[¶](https://docs.nextcloud.com/server/14/developer_manual/app/api.html#modifying-the-cors-headers "Permalink to this headline")
By default the following values will be used for the preflighted OPTIONS request:
  * **Access-Control-Allow-Methods** : ‘PUT, POST, GET, DELETE, PATCH’
  * **Access-Control-Allow-Headers** : ‘Authorization, Content-Type, Accept’
  * **Access-Control-Max-Age** : 1728000


To add an additional method or header or allow less headers, simply pass additional values to the parent constructor:
```
<?php
namespace OCA\MyApp\Controller;

use \OCP\AppFramework\ApiController;
use \OCP\IRequest;

class AuthorApiController extends ApiController {

    public function __construct($appName, IRequest $request) {
        parent::__construct(
            $appName,
            $request,
            'PUT, POST, GET, DELETE, PATCH',
            'Authorization, Content-Type, Accept',
            1728000);
    }

}

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html "Templates") [](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html "Controllers")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
