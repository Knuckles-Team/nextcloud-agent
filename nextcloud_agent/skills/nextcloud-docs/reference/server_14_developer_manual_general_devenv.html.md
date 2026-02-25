[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
    * [Community code of conduct](https://docs.nextcloud.com/server/14/developer_manual/general/code-of-conduct.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html)
      * [Set up Web server and database](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#set-up-web-server-and-database)
      * [](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#get-the-source)
        * [Gather information about server setup](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#gather-information-about-server-setup)
        * [Check out the code](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#check-out-the-code)
        * [Enabling debug mode](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#enabling-debug-mode)
        * [Keep the code up-to-date](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#keep-the-code-up-to-date)
    * [Security guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/security.html)
    * [Coding style & general guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/codingguidelines.html)
    * [Performance considerations](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html)
    * [Debugging](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html)
    * [Backporting](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html)
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
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [General contributor guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/index.html) »
  * Development environment
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/general/devenv.rst)


* * *
# Development environment[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#development-environment "Permalink to this headline")
Please follow the steps on this page to set up your development environment.
## Set up Web server and database[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#set-up-web-server-and-database "Permalink to this headline")
First [set up your Web server and database](https://docs.nextcloud.org/server/14/admin_manual/installation/index.html) (**Section** : Manual Installation - Prerequisites).
## Get the source[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#get-the-source "Permalink to this headline")
There are two ways to obtain Nextcloud sources:
  * Using the [stable version](https://docs.nextcloud.org/server/14/admin_manual/#installation)
  * Using the development version from [GitHub](https://github.com/nextcloud) which will be explained below.


To check out the source from [GitHub](https://github.com/nextcloud) you will need to install Git (see [Setting up Git](https://help.github.com/articles/set-up-git) from the GitHub help)
### Gather information about server setup[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#gather-information-about-server-setup "Permalink to this headline")
To get started the basic Git repositories need to be cloned into the Web server’s directory. Depending on the distribution this will either be
  * **/var/www**
  * **/var/www/html**
  * **/srv/http**


Then identify the user and group the Web server is running as and the Apache user and group for the **chown** command will either be
  * **http**
  * **www-data**
  * **apache**
  * **wwwrun**


### Check out the code[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#check-out-the-code "Permalink to this headline")
The following commands are using **/var/www** as the Web server’s directory and **www-data** as user name and group.
After the development tool installation make the directory writable so you install the code as your regular user, and don’t need root privileges:
```
sudo chmod o+rw /var/www

```

Then install Nextcloud at the root of your site from Git:
```
git clone https://github.com/nextcloud/server.git /var/www/
cd /var/www
git submodule update --init

```

If you like to install Nextcloud at a sub-folder, replace /var/www with /var/www/<folder>.
Create the data and the config folders:
```
cd /var/www
mkdir data
mkdir config

```

Adjust rights:
```
cd /var/www
sudo chown -R www-data:www-data config data apps
sudo chmod o-rw /var/www

```

Finally restart the Web server (this might vary depending on your distribution):
```
sudo systemctl restart httpd.service

```

or:
```
sudo systemctl restart apache2.service

```

or:
```
sudo /etc/init.d/apache2 restart

```

Now access the installation at <http://localhost/> (or the corresponding URL) in your web browser to set up your instance.
### Enabling debug mode[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#enabling-debug-mode "Permalink to this headline")
Note
Do not enable this for production! This can create security problems and is only meant for debugging and development!
To disable JavaScript and CSS caching debugging has to be enabled by setting `debug` to `true` in `config/config.php`:
```
<?php
$CONFIG = array (
    'debug' => true,
    ... configuration goes here ...
);

```

### Keep the code up-to-date[¶](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html#keep-the-code-up-to-date "Permalink to this headline")
If you have more than one repository cloned, it can be time consuming to do the same the action to all repositories one by one. To solve this, you can use the following command template:
```
find . -maxdepth <DEPTH> -type d -name .git -exec sh -c 'cd "{}"/../ && pwd && <GIT COMMAND>' \;

```

then, e.g. to pull all changes in all repositories, you only need this:
```
find . -maxdepth 3 -type d -name .git -exec sh -c 'cd "{}"/../ && pwd && git pull --rebase' \;

```

or to prune all merged branches, you would execute this:
```
find . -maxdepth 3 -type d -name .git -exec sh -c 'cd "{}"/../ && pwd && git remote prune origin' \;

```

It is even easier if you create alias from these commands in case you want to avoid retyping those each time you need them.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/general/security.html "Security guidelines") [](https://docs.nextcloud.com/server/14/developer_manual/general/code-of-conduct.html "Community code of conduct")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
