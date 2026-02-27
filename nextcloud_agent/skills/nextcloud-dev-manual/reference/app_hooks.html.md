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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#available-hooks)
      * [Session](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#session)
      * [UserManager](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#usermanager)
      * [GroupManager](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#groupmanager)
      * [Filesystem root](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#filesystem-root)
      * [Filesystem scanner](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#filesystem-scanner)
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
  * Hooks
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/hooks.rst)


* * *
# Hooks[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#hooks "Permalink to this headline")
Hooks are used to execute code before or after an event has occurred. This is for instance useful to run cleanup code after users, groups or files have been deleted. Hooks should be registered in the [app.php](https://docs.nextcloud.com/server/14/developer_manual/app/init.html):
```
<?php
namespace OCA\MyApp\AppInfo;

$app = new Application();
$app->getContainer()->query('UserHooks')->register();

```

The hook logic should be in a separate class that is being registered in the [Container](https://docs.nextcloud.com/server/14/developer_manual/app/container.html):
```
<?php
namespace OCA\MyApp\AppInfo;

use \OCP\AppFramework\App;

use \OCA\MyApp\Hooks\UserHooks;


class Application extends App {

    public function __construct(array $urlParams=array()){
        parent::__construct('myapp', $urlParams);

        $container = $this->getContainer();

        /**
         * Controllers
         */
        $container->registerService('UserHooks', function($c) {
            return new UserHooks(
                $c->query('ServerContainer')->getUserManager()
            );
        });
    }
}

```

```
<?php
namespace OCA\MyApp\Hooks;
use OCP\IUserManager;

class UserHooks {

    private $userManager;

    public function __construct(IUserManager $userManager){
        $this->userManager = $userManager;
    }

    public function register() {
        $callback = function($user) {
            // your code that executes before $user is deleted
        };
        $this->userManager->listen('\OC\User', 'preDelete', $callback);
    }

}

```

## Available hooks[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#available-hooks "Permalink to this headline")
The scope is the first parameter that is passed to the **listen** method, the second parameter is the method and the third one the callback that should be executed once the hook is being called, e.g.:
```
<?php

// listen on user predelete
$callback = function($user) {
    // your code that executes before $user is deleted
};
$userManager->listen('\OC\User', 'preDelete', $callback);

```

Hooks can also be removed by using the **removeListener** method on the object:
```
<?php

// delete previous callback
$userManager->removeListener(null, null, $callback);

```

The following hooks are available:
### Session[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#session "Permalink to this headline")
Injectable from the ServerContainer by calling the method **getUserSession()**.
Hooks available in scope **\OC\User** :
  * **preSetPassword** (\OC\User\User $user, string $password, string $recoverPassword)
  * **postSetPassword** (\OC\User\User $user, string $password, string $recoverPassword)
  * **changeUser** (\OC\User\User $user, string $feature, string $value)
  * **preDelete** (\OC\User\User $user)
  * **postDelete** (\OC\User\User $user)
  * **preCreateUser** (string $uid, string $password)
  * **postCreateUser** (\OC\User\User $user)
  * **preLogin** (string $user, string $password)
  * **postLogin** (\OC\User\User $user, string $password)
  * **logout** ()


### UserManager[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#usermanager "Permalink to this headline")
Injectable from the ServerContainer by calling the method **getUserManager()**.
Hooks available in scope **\OC\User** :
  * **preSetPassword** (\OC\User\User $user, string $password, string $recoverPassword)
  * **postSetPassword** (\OC\User\User $user, string $password, string $recoverPassword)
  * **preDelete** (\OC\User\User $user)
  * **postDelete** (\OC\User\User $user)
  * **preCreateUser** (string $uid, string $password)
  * **postCreateUser** (\OC\User\User $user, string $password)


### GroupManager[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#groupmanager "Permalink to this headline")
Hooks available in scope **\OC\Group** :
  * **preAddUser** (\OC\Group\Group $group, \OC\User\User $user)
  * **postAddUser** (\OC\Group\Group $group, \OC\User\User $user)
  * **preRemoveUser** (\OC\Group\Group $group, \OC\User\User $user)
  * **postRemoveUser** (\OC\Group\Group $group, \OC\User\User $user)
  * **preDelete** (\OC\Group\Group $group)
  * **postDelete** (\OC\Group\Group $group)
  * **preCreate** (string $groupId)
  * **postCreate** (\OC\Group\Group $group)


### Filesystem root[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#filesystem-root "Permalink to this headline")
Injectable from the ServerContainer by calling the method **getRootFolder()** , **getUserFolder()** or **getAppFolder()**.
Filesystem hooks available in scope **\OC\Files** :
  * **preWrite** (\OCP\Files\Node $node)
  * **postWrite** (\OCP\Files\Node $node)
  * **preCreate** (\OCP\Files\Node $node)
  * **postCreate** (\OCP\Files\Node $node)
  * **preDelete** (\OCP\Files\Node $node)
  * **postDelete** (\OCP\Files\Node $node)
  * **preTouch** (\OCP\Files\Node $node, int $mtime)
  * **postTouch** (\OCP\Files\Node $node)
  * **preCopy** (\OCP\Files\Node $source, \OCP\Files\Node $target)
  * **postCopy** (\OCP\Files\Node $source, \OCP\Files\Node $target)
  * **preRename** (\OCP\Files\Node $source, \OCP\Files\Node $target)
  * **postRename** (\OCP\Files\Node $source, \OCP\Files\Node $target)


### Filesystem scanner[¶](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html#filesystem-scanner "Permalink to this headline")
Filesystem scanner hooks available in scope **\OC\Files\Utils\Scanner** :
  * **scanFile** (string $absolutePath)
  * **scanFolder** (string $absolutePath)
  * **postScanFile** (string $absolutePath)
  * **postScanFolder** (string $absolutePath)


[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/backgroundjobs.html "Background jobs \(Cron\)") [](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html "Two-factor providers")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
