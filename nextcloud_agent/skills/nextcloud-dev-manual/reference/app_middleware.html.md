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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html)
    * [Parsing annotations](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html#parsing-annotations)
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
  * Middleware
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/middleware.rst)


* * *
# Middleware[¶](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html#middleware "Permalink to this headline")
Middleware is logic that is run before and after each request and is modelled after [Django’s Middleware system](https://docs.djangoproject.com/en/dev/topics/http/middleware/). It offers the following hooks:
  * **beforeController** : This is executed before a controller method is being executed. This allows you to plug additional checks or logic before that method, like for instance security checks
  * **afterException** : This is being run when either the beforeController method or the controller method itself is throwing an exception. The middleware is asked in reverse order to handle the exception and to return a response. If the middleware can’t handle the exception, it throws the exception again
  * **afterController** : This is being run after a successful controller method call and allows the manipulation of a Response object. The middleware is run in reverse order
  * **beforeOutput** : This is being run after the response object has been rendered and allows the manipulation of the outputted text. The middleware is run in reverse order


To generate your own middleware, simply inherit from the Middleware class and overwrite the methods that should be used.
```
<?php

namespace OCA\MyApp\Middleware;

use \OCP\AppFramework\Middleware;


class CensorMiddleware extends Middleware {

    /**
     * this replaces "bad words" with "********" in the output
     */
    public function beforeOutput($controller, $methodName, $output){
        return str_replace('bad words', '********', $output);
    }

}

```

The middleware can be registered in the [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html) and added using the **registerMiddleware** method:
```
<?php

namespace OCA\MyApp\AppInfo;

use \OCP\AppFramework\App;

use \OCA\MyApp\Middleware\CensorMiddleware;

class MyApp extends App {

    /**
     * Define your dependencies in here
     */
    public function __construct(array $urlParams=array()){
        parent::__construct('myapp', $urlParams);

        $container = $this->getContainer();

        /**
         * Middleware
         */
        $container->registerService('CensorMiddleware', function($c){
            return new CensorMiddleware();
        });

        // executed in the order that it is registered
        $container->registerMiddleware('CensorMiddleware');

    }
}

```

Note
The order is important! The middleware that is registered first gets run first in the **beforeController** method. For all other hooks, the order is being reversed, meaning: if a middleware is registered first, it gets run last.
## Parsing annotations[¶](https://docs.nextcloud.com/server/14/developer_manual/app/middleware.html#parsing-annotations "Permalink to this headline")
Sometimes it is useful to conditionally execute code before or after a controller method. This can be done by defining custom annotations. An example would be to add a custom authentication method or simply add an additional header to the response. To access the parsed annotations, inject the **ControllerMethodReflector** class:
```
<?php

namespace OCA\MyApp\Middleware;

use \OCP\AppFramework\Middleware;
use \OCP\AppFramework\Utility\ControllerMethodReflector;
use \OCP\IRequest;

class HeaderMiddleware extends Middleware {

  private $reflector;

  public function __construct(ControllerMethodReflector $reflector) {
      $this->reflector = $reflector;
  }

  /**
   * Add custom header if @MyHeader is used
   */
  public function afterController($controller, $methodName, IResponse $response){
      if($this->reflector->hasAnnotation('MyHeader')) {
          $response->addHeader('My-Header', 3);
      }
      return $response;
  }

}

```

Now adjust the container to inject the reflector:
```
<?php

namespace OCA\MyApp\AppInfo;

use \OCP\AppFramework\App;

use \OCA\MyApp\Middleware\HeaderMiddleware;

class MyApp extends App {

    /**
     * Define your dependencies in here
     */
    public function __construct(array $urlParams=array()){
        parent::__construct('myapp', $urlParams);

        $container = $this->getContainer();

        /**
         * Middleware
         */
        $container->registerService('HeaderMiddleware', function($c){
            return new HeaderMiddleware($c->query('ControllerMethodReflector'));
        });

        // executed in the order that it is registered
        $container->registerMiddleware('HeaderMiddleware');
    }

}

```

Note
An annotation always starts with an uppercase letter
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/container.html "Container") [](https://docs.nextcloud.com/server/14/developer_manual/app/routes.html "Routing")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
