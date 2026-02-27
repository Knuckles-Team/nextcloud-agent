[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
  * [Changelog](https://docs.nextcloud.com/server/14/developer_manual/app/changelog.html)
  * [Tutorial](https://docs.nextcloud.com/server/14/developer_manual/app/tutorial.html)
  * [Create an app](https://docs.nextcloud.com/server/14/developer_manual/app/startapp.html)
  * [Navigation and pre-app configuration](https://docs.nextcloud.com/server/14/developer_manual/app/init.html)
  * [App metadata](https://docs.nextcloud.com/server/14/developer_manual/app/info.html)
  * [Classloader](https://docs.nextcloud.com/server/14/developer_manual/app/classloader.html)
  * [Request lifecycle](https://docs.nextcloud.com/server/14/developer_manual/app/request.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html)
    * [Extracting values from the URL](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#extracting-values-from-the-url)
    * [Matching subURLs](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#matching-suburls)
    * [Default values for subURL](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#default-values-for-suburl)
    * [Registering resources](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#registering-resources)
    * [Using the URLGenerator](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#using-the-urlgenerator)
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
  * Routing
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/routes.rst)


* * *
# Routing[¶](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#routing "Permalink to this headline")
Routes map a URL and a method to a controller method. Routes are defined inside `appinfo/routes.php` by returning them as an array:
```
<?php
return [
    'routes' => [
        ['name' => 'page#index', 'url' => '/', 'verb' => 'GET'],
    ],
];

```

The route array contains the following parts:
  * **url** : The URL that is matched after _/index.php/apps/myapp_
  * **name** : The controller and the method to call; _page#index_ is being mapped to _PageController- >index()_, _articles_api#drop_latest_ would be mapped to _ArticlesApiController- >dropLatest()_. The controller in the example above would be stored in `lib/Controller/PageController.php`.
  * **method** (Optional, defaults to GET): The HTTP method that should be matched, (e.g. GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH)
  * **requirements** (Optional): lets you match and extract URLs that have slashes in them (see [Matching subURLs](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#matching-suburls))
  * **postfix** (Optional): lets you define a route id postfix. Since each route name will be transformed to a route id (**page#method** -> **myapp.page.method**) and the route id can only exist once you can use the postfix option to alter the route id creation by adding a string to the route id, e.g., **‘name’ = > ‘page#method’, ‘postfix’ => ‘test’** will yield the route id **myapp.page.methodtest**. This makes it possible to add more than one route/URL for a controller method
  * **defaults** (Optional): If this setting is given, a default value will be assumed for each URL parameter which is not present. The default values are passed in as a key => value par array


## Extracting values from the URL[¶](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#extracting-values-from-the-url "Permalink to this headline")
It is possible to extract values from the URL to allow RESTful URL design. To extract a value, you have to wrap it inside curly braces:
```
<?php

// Request: GET /index.php/apps/myapp/authors/3

// appinfo/routes.php
array('name' => 'author#show', 'url' => '/authors/{id}', 'verb' => 'GET'),

// controller/authorcontroller.php
class AuthorController {

    public function show($id) {
        // $id is '3'
    }

}

```

The identifier used inside the route is being passed into controller method by reflecting the method parameters. So basically if you want to get the value **{id}** in your method, you need to add **$id** to your method parameters.
## Matching subURLs[¶](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#matching-suburls "Permalink to this headline")
Sometimes it is needed to match more than one URL fragment. An example would be to match a request for all URLs that start with **OPTIONS /index.php/apps/myapp/api**. To do this, use the **requirements** parameter in your route which is an array containing pairs of **‘key’ = > ‘regex’**:
```
<?php

// Request: OPTIONS /index.php/apps/myapp/api/my/route

// appinfo/routes.php
array('name' => 'author_api#cors', 'url' => '/api/{path}', 'verb' => 'OPTIONS',
      'requirements' => array('path' => '.+')),

// controller/authorapicontroller.php
class AuthorApiController {

    public function cors($path) {
        // $path will be 'my/route'
    }

}

```

## Default values for subURL[¶](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#default-values-for-suburl "Permalink to this headline")
Apart from matching requirements, a subURL may also have a default value. Say you want to support pagination (a ‘page’ parameter) for your **/posts** subURL that displays posts entries list. You may set a default value for the ‘page’ parameter, that will be used if not already set in the URL. Use the **defaults** parameter in your route which is an array containing pairs of **‘urlparameter’ = > ‘defaultvalue’**:
```
<?php

// Request: GET /index.php/app/myapp/post

// appinfo/routes.php
array(
    'name'     => 'post#index',
    'url'      => '/post/{page}',
    'verb'     => 'GET',
    'defaults' => array('page' => 1) // this allows same URL as /index.php/myapp/post/1
),

// controller/postcontroller.php
class PostController
{
    public function index($page = 1)
    {
        // $page will be 1
    }
}

```

## Registering resources[¶](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#registering-resources "Permalink to this headline")
When dealing with resources, writing routes can become quite repetitive since most of the time routes for the following tasks are needed:
  * Get all entries
  * Get one entry by id
  * Create an entry
  * Update an entry
  * Delete an entry


To prevent repetition, it’s possible to define resources. The following routes:
```
<?php
return [
    'routes' => [
        ['name' => 'author#index', 'url' => '/authors', 'verb' => 'GET'],
        ['name' => 'author#show', 'url' => '/authors/{id}', 'verb' => 'GET'],
        ['name' => 'author#create', 'url' => '/authors', 'verb' => 'POST'],
        ['name' => 'author#update', 'url' => '/authors/{id}', 'verb' => 'PUT'],
        ['name' => 'author#destroy', 'url' => '/authors/{id}', 'verb' => 'DELETE'],
        // your other routes here
    ],
];

```

can be abbreviated by using the **resources** key:
```
<?php
return [
    'resources' => [
        'author' => [url' => '/authors'],
    ],
    'routes' => [
        // your other routes here
    ],
];

```

## Using the URLGenerator[¶](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html#using-the-urlgenerator "Permalink to this headline")
Sometimes it is useful to turn a route into a URL to make the code independent from the URL design or to generate a URL for an image in **img/**. Inside the PageController the URL generator can be injected by adding it to the constructor, which will allow to use it to generate a URL for a redirect. For more details on that see the [Dependency injection](https://docs.nextcloud.com/server/14/developer_manual/app/container.html#id1) reference.
```
<?php
namespace OCA\MyApp\Controller;

use \OCP\IRequest;
use \OCP\IURLGenerator;
use \OCP\AppFramework\Controller;
use \OCP\AppFramework\Http\RedirectResponse;

class PageController extends Controller {

    private $urlGenerator;

    public function __construct($appName, IRequest $request,
                                IURLGenerator $urlGenerator) {
        parent::__construct($appName, $request);
        $this->urlGenerator = $urlGenerator;
    }

    /**
     * redirect to /apps/news/myapp/authors/3
     */
    public function redirect() {
        // route name: author_api#do_something
        // route url: /apps/news/myapp/authors/{id}

        // # needs to be replaced with a . due to limitations and prefixed
        // with your app id
        $route = 'myapp.author_api.do_something';
        $parameters = array('id' => 3);

        $url = $this->urlGenerator->linkToRoute($route, $parameters);

        return new RedirectResponse($url);
    }

}

```

URLGenerator is case sensitive, so **appName** must match **exactly** the name you use in [configuration](https://docs.nextcloud.com/server/14/developer_manual/app/configuration.html). If you use a CamelCase name as _myCamelCaseApp_ ,
```
<?php
$route = 'myCamelCaseApp.author_api.do_something';

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html "Middleware") [](https://docs.nextcloud.com/server/14/developer_manual/app/request.html "Request lifecycle")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
