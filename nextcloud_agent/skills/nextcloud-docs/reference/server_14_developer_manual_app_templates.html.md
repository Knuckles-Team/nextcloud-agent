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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html)
    * [Including templates](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#including-templates)
    * [Including CSS and JavaScript](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#including-css-and-javascript)
    * [Including images](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#including-images)
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
  * Templates
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/templates.rst)


* * *
# Templates[¶](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#templates "Permalink to this headline")
Nextcloud provides its own templating system which is basically plain PHP with some additional functions and preset variables. All the parameters which have been passed from the [controller](https://docs.nextcloud.com/server/14/developer_manual/app/controllers.html) are available in an array called **$_[]** , e.g.:
```
array('key' => 'something')

```

can be accessed through:
```
$_['key']

```

Note
To prevent XSS the following PHP **functions for printing are forbidden: echo, print() and <?=**. Instead use the **p()** function for printing your values. Should you require unescaped printing, **double check for XSS** and use: [`print_unescaped`](https://docs.nextcloud.com/server/14/developer_manual/api/index.html#print_unescaped "print_unescaped").
Printing values is done by using the **p()** function, printing HTML is done by using **print_unescaped()**
`templates/main.php`
```
<?php foreach($_['entries'] as $entry){ ?>
  <p><?php p($entry); ?></p>
<?php
}

```

## Including templates[¶](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#including-templates "Permalink to this headline")
Templates can also include other templates by using the **$this- >inc(‘templateName’)** method.
```
<?php print_unescaped($this->inc('sub.inc')); ?>

```

The parent variables will also be available in the included templates, but should you require it, you can also pass new variables to it by using the second optional parameter as array for **$this- >inc**.
`templates/sub.inc.php`
```
<div>I am included, but I can still access the parents variables!</div>
<?php p($_['name']); ?>

<?php print_unescaped($this->inc('other_template', array('variable' => 'value'))); ?>

```

## Including CSS and JavaScript[¶](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#including-css-and-javascript "Permalink to this headline")
To include CSS or JavaScript use the **style** and **script** functions:
```
<?php
script('myapp', 'script');  // add js/script.js
style('myapp', 'style');  // add css/style.css

```

## Including images[¶](https://docs.nextcloud.com/server/14/developer_manual/app/templates.html#including-images "Permalink to this headline")
To generate links to images use the **image_path** function:
```
<img src="<?php print_unescaped(image_path('myapp', 'app.png')); ?>" />

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/js.html "JavaScript") [](https://docs.nextcloud.com/server/14/developer_manual/app/api.html "RESTful API")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
