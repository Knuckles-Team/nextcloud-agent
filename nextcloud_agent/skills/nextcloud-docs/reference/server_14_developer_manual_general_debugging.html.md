[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
    * [Community code of conduct](https://docs.nextcloud.com/server/14/developer_manual/general/code-of-conduct.html)
    * [Development environment](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html)
    * [Security guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/security.html)
    * [Coding style & general guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/codingguidelines.html)
    * [Performance considerations](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html)
      * [Debug mode](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debug-mode)
      * [Identifying errors](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#identifying-errors)
      * [Debugging variables](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging-variables)
      * [Using a PHP debugger (XDebug)](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#using-a-php-debugger-xdebug)
      * [Debugging JavaScript](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging-javascript)
      * [Debugging HTML and templates](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging-html-and-templates)
      * [Using alternative app directories](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#using-alternative-app-directories)
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
  * Debugging
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/general/debugging.rst)


* * *
# Debugging[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging "Permalink to this headline")
## Debug mode[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debug-mode "Permalink to this headline")
When debug mode is enabled in Nextcloud, a variety of debugging features are enabled - see debugging documentation. Set `debug` to `true` in `/config/config.php` to enable it:
```
<?php
$CONFIG = array (
    'debug' => true,
    ... configuration goes here ...
);

```

## Identifying errors[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#identifying-errors "Permalink to this headline")
Nextcloud uses custom error PHP handling that prevents errors being printed to Web server log files or command line output. Instead, errors are generally stored in Nextcloud’s own log file, located at: `/data/nextcloud.log`
## Debugging variables[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging-variables "Permalink to this headline")
You should use exceptions if you need to debug variable values manually, and not alternatives like trigger_error() (which may not be logged).
e.g.:
```
<?php throw new \Exception( "\$user = $user" ); // should be logged in Nextcloud ?>

```

not:
```
<?php trigger_error( "\$user = $user" ); // may not be logged anywhere ?>

```

To disable custom error handling in Nextcloud (and have PHP and your Web server handle errors instead), see Debug mode.
## Using a PHP debugger (XDebug)[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#using-a-php-debugger-xdebug "Permalink to this headline")
Using a debugger connected to PHP allows you to step through code line by line, view variables at each line and even change values while the code is running. The de-facto standard debugger for PHP is XDebug, available as an installable package in many distributions. It just provides the PHP side however, so you will need a frontend to actually control XDebug. When installed, it needs to be enabled in `php.ini`, along with some parameters to enable connections to the debugging interface:
```
zend_extension=/usr/lib/php/modules/xdebug.so
xdebug.remote_enable=on
xdebug.remote_host=127.0.0.1
xdebug.remote_port=9000
xdebug.remote_handler=dbgp

```

XDebug will now (when activated) try to connect to localhost on port 9000, and will communicate over the standard protocol DBGP. This protocol is supported by many debugging interfaces, such as the following popular ones:
  * vdebug - Multi-language DBGP debugger client for Vim
  * SublimeTextXdebug - XDebug client for Sublime Text
  * PHPStorm - in-built DBGP debugger


For further reading, see the XDebug documentation: <http://xdebug.org/docs/remote>
Once you are familiar with how your debugging client works, you can start debugging with XDebug. To test Nextcloud through the web interface or other HTTP requests, set the `XDEBUG_SESSION_START` cookie or POST parameter. Alternatively, there are browser extensions to make this easy:
  * The Easiest XDebug (Firefox): <https://addons.mozilla.org/en-US/firefox/addon/the-easiest-xdebug/>
  * XDebug Helper (Chrome): <https://chrome.google.com/extensions/detail/eadndfjplgieldjbigjakmdgkmoaaaoc>


For debugging scripts on the command line, like `occ` or unit tests, set the `XDEBUG_CONFIG` environment variable.
## Debugging JavaScript[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging-javascript "Permalink to this headline")
By default all JavaScript files in Nextcloud are minified (compressed) into a single file without whitespace. To prevent this, see Debug mode.
## Debugging HTML and templates[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#debugging-html-and-templates "Permalink to this headline")
By default Nextcloud caches HTML generated by templates. This may prevent changes to app templates, for example, from being applied on page refresh. To disable caching, see Debug mode.
## Using alternative app directories[¶](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html#using-alternative-app-directories "Permalink to this headline")
It may be useful to have multiple app directories for testing purposes, so you can conveniently switch between different versions of applications. See the configuration file documentation for details.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/general/backporting.html "Backporting") [](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html "Performance considerations")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
