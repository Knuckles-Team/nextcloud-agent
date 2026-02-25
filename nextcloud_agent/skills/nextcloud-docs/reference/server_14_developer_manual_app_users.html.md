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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/users.html)
    * [Creating users](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#creating-users)
    * [Modifying users](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#modifying-users)
    * [User session information](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#user-session-information)
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
  * User management
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/users.rst)


* * *
# User management[¶](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#user-management "Permalink to this headline")
Users can be managed using the UserManager which is injected from the ServerContainer:
```
<?php
namespace OCA\MyApp\AppInfo;

use \OCP\AppFramework\App;

use \OCA\MyApp\Service\UserService;


class Application extends App {

    public function __construct(array $urlParams=array()){
        parent::__construct('myapp', $urlParams);

        $container = $this->getContainer();

        /**
         * Controllers
         */
        $container->registerService('UserService', function($c) {
            return new UserService(
                $c->query('UserManager')
            );
        });

        $container->registerService('UserManager', function($c) {
            return $c->query('ServerContainer')->getUserManager();
        });
    }
}

```

## Creating users[¶](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#creating-users "Permalink to this headline")
Creating a user is done by passing a username and password to the create method:
```
<?php
namespace OCA\MyApp\Service;

class UserService {

    private $userManager;

    public function __construct($userManager){
        $this->userManager = $userManager;
    }

    public function create($userId, $password) {
        return $this->userManager->create($userId, $password);
    }

}

```

## Modifying users[¶](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#modifying-users "Permalink to this headline")
Users can be modified by getting a user by the userId or by a search pattern. The returned user objects can then be used to:
  * Delete them
  * Set a new password
  * Disable/Enable them
  * Get their home directory


```
<?php
namespace OCA\MyApp\Service;

class UserService {

    private $userManager;

    public function __construct($userManager){
        $this->userManager = $userManager;
    }

    public function delete($userId) {
        return $this->userManager->get($userId)->delete();
    }

    // recoveryPassword is used for the encryption app to recover the keys
    public function setPassword($userId, $password, $recoveryPassword) {
        return $this->userManager->get($userId)->setPassword($password, $recoveryPassword);
    }

    public function disable($userId) {
        return $this->userManager->get($userId)->setEnabled(false);
    }

    public function getHome($userId) {
        return $this->userManager->get($userId)->getHome();
    }
}

```

## User session information[¶](https://docs.nextcloud.com/server/14/developer_manual/app/users.html#user-session-information "Permalink to this headline")
To login, logout or getting the currently logged in user, the UserSession has to be injected from the ServerContainer:
```
<?php
namespace OCA\MyApp\AppInfo;

use \OCP\AppFramework\App;

use \OCA\MyApp\Service\UserService;


class Application extends App {

    public function __construct(array $urlParams=array()){
        parent::__construct('myapp', $urlParams);

        $container = $this->getContainer();

        /**
         * Controllers
         */
        $container->registerService('UserService', function($c) {
            return new UserService(
                $c->query('UserSession')
            );
        });

        $container->registerService('UserSession', function($c) {
            return $c->query('ServerContainer')->getUserSession();
        });

        // currently logged in user, userId can be gotten by calling the
        // getUID() method on it
        $container->registerService('User', function($c) {
            return $c->query('UserSession')->getUser();
        });
    }
}

```

Then users can be logged in by using:
```
<?php
namespace OCA\MyApp\Service;

class UserService {

    private $userSession;

    public function __construct($userSession){
        $this->userSession = $userSession;
    }

    public function login($userId, $password) {
        return $this->userSession->login($userId, $password);
    }

    public function logout() {
        $this->userSession->logout();
    }

}

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html "Two-factor providers") [](https://docs.nextcloud.com/server/14/developer_manual/app/appdata.html "AppData")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
