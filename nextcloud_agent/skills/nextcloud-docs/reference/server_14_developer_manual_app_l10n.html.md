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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html)
    * [PHP](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#php)
    * [Templates](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#templates)
    * [JavaScript](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#javascript)
    * [Hints](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#hints)
    * [Ignoring files from translation tool](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#ignoring-files-from-translation-tool)
    * [Creating your own translatable files](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#creating-your-own-translatable-files)
    * [Setup of the transifex sync](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#setup-of-the-transifex-sync)
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
  * Translation
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/l10n.rst)


* * *
# Translation[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#translation "Permalink to this headline")
Nextcloud’s translation system is powered by [Transifex](https://www.transifex.com/nextcloud/). To start translating sign up and enter a group. If your community app should be translated by the [Nextcloud community on Transifex](https://www.transifex.com/nextcloud/nextcloud/dashboard/) just follow the setup section below.
## PHP[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#php "Permalink to this headline")
Should it ever be needed to use localized strings on the server-side, simply inject the L10N service from the ServerContainer into the needed constructor:
```
<?php
namespace OCA\MyApp\AppInfo;

use \OCP\AppFramework\App;

use \OCA\MyApp\Service\AuthorService;


class Application extends App {

    public function __construct(array $urlParams=array()){
        parent::__construct('myapp', $urlParams);

        $container = $this->getContainer();

        /**
         * Controllers
         */
        $container->registerService('AuthorService', function($c) {
            return new AuthorService(
                $c->query('L10N')
            );
        });

        $container->registerService('L10N', function($c) {
            return $c->query('ServerContainer')->getL10N($c->query('AppName'));
        });
    }
}

```

Strings can then be translated in the following way:
```
<?php
namespace OCA\MyApp\Service;

use \OCP\IL10N;


class AuthorService {

    private $trans;

    public function __construct(IL10N $trans){
        $this->trans = $trans;
    }

    public function getLanguageCode() {
        return $this->trans->getLanguageCode();
    }

    public sayHello() {
        return $this->trans->t('Hello');
    }

    public function getAuthorName($name) {
        return $this->trans->t('Getting author %s', array($name));
    }

    public function getAuthors($count, $city) {
        return $this->trans->n(
            '%n author is currently in the city %s',  // singular string
            '%n authors are currently in the city %s',  // plural string
            $count,
            array($city)
        );
    }
}

```

## Templates[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#templates "Permalink to this headline")
In every template the global variable **$l** can be used to translate the strings using its methods **t()** and **n()** :
```
<div><?php p($l->t('Showing %s files', $_['count'])); ?></div>

<button><?php p($l->t('Hide')); ?></button>

```

## JavaScript[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#javascript "Permalink to this headline")
There is a global function **t()** available for translating strings. The first argument is your app name, the second argument is the string to translate.
```
t('myapp', 'Hello World!');

```

For advanced usage, refer to the source code **core/js/l10n.js** ; **t()** is bind to **OC.L10N.translate()**.
## Hints[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#hints "Permalink to this headline")
In case some translation strings may be translated wrongly because they have multiple meanings, you can add hints which will be shown in the Transifex web-interface:
```
<ul id="translations">
    <li id="add-new">
        <?php
            // TRANSLATORS Will be shown inside a popup and asks the user to add a new file
            p($l->t('Add new file'));
        ?>
    </li>
</ul>

```

## Ignoring files from translation tool[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#ignoring-files-from-translation-tool "Permalink to this headline")
The translation tool scrapes the source code for method calls to **t()** or **n()** to extract the strings that should be translated. If you check in minified JS code for example then those method names are also quite common and could cause wrong extractions. For this reason we allow to specify a list of files that the translation tool will not scrape for strings. You simply need to add a file named `.l10nignore` into the root folder of your app and specify the files one per line:
```
# compiled vue templates
js/bruteforcesettings.js

```

## Creating your own translatable files[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#creating-your-own-translatable-files "Permalink to this headline")
If Transifex is not the right choice or the app is not accepted for translation, generate the gettext strings by yourself by executing our [translation tool](https://github.com/nextcloud/docker-ci/tree/master/translations/translationtool) in the app folder:
```
cd /srv/http/nextcloud/apps/myapp
translationtool.phar create-pot-files

```

The translation tool requires **gettext** , installable via:
```
apt-get install gettext

```

The above tool generates a template that can be used to translate all strings of an app. This template is located in the folder `translationfiles/template/` with the name `myapp.pot`. It can be used by your favored translation tool which then creates a `.po` file. The `.po` file needs to be placed in a folder named like the language code with the app name as filename - for example `translationfiles/es/myapp.po`. After this step the tool needs to be invoked to transfer the po file into our own fileformat that is more easily readable by the server code:
```
translationtool.phar convert-po-files

```

Now the following folder structure is available:
```
myapp/l10n
|-- es.js
|-- es.json
myapp/translationfiles
|-- es
|   |-- myapp.po
|-- templates
    |-- myapp.pot

```

You then just need the `.json` and `.js` files for a working localized app.
## Setup of the transifex sync[¶](https://docs.nextcloud.com/server/14/developer_manual/app/l10n.html#setup-of-the-transifex-sync "Permalink to this headline")
To setup the transifex sync within the Nextcloud community you need to add first the transifex config to your app folder at `.tx/config` (please replace **MYAPP** with your apps id):
```
[main]
host = https://www.transifex.com
lang_map = bg_BG: bg, cs_CZ: cs, fi_FI: fi, hu_HU: hu, nb_NO: nb, sk_SK: sk, th_TH: th, ja_JP: ja

[nextcloud.MYAPP]
file_filter = translationfiles/<lang>/MYAPP.po
source_file = translationfiles/templates/MYAPP.pot
source_lang = en
type = PO

```

Then create a folder `l10n` and a file `l10n/.gitkeep` to create an empty folder which later holds the translations.
Now the GitHub account [@nextcloud-bot](https://github.com/nextcloud-bot) needs to get write access to your repository. It will run every night and only push commits to the master branch once there is an update to the translation. In general you should enable the [protected branches feature](https://help.github.com/articles/configuring-protected-branches/) at least for the master branch.
For the sync job there is a [configuration file](https://github.com/nextcloud/docker-ci/blob/master/translations/config.json) available in our docker-ci repository. Adding there the repo owner and repo name to the section named **app** via pull request is enough. Once this change is in one member of the sysadmin team will deploy it to the sync server and the job will then run once a day.
If you need help then just [open a ticket with the request](https://github.com/nextcloud/docker-ci/issues/new) and we can also guide you through the steps.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/theming.html "Theming support") [](https://docs.nextcloud.com/server/14/developer_manual/app/css.html "CSS")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
