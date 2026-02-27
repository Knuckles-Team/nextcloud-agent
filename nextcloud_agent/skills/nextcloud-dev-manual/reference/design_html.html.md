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
  * [](https://docs.nextcloud.com/server/14/developer_manual/design/index.html)
    * [Introduction](https://docs.nextcloud.com/server/14/developer_manual/design/navigation.html)
    * [New button](https://docs.nextcloud.com/server/14/developer_manual/design/navigation.html#new-button)
    * [App navigation menu](https://docs.nextcloud.com/server/14/developer_manual/design/navigation.html#app-navigation-menu)
    * [Settings](https://docs.nextcloud.com/server/14/developer_manual/design/navigation.html#settings)
    * [Main content](https://docs.nextcloud.com/server/14/developer_manual/design/content.html)
    * [Content list](https://docs.nextcloud.com/server/14/developer_manual/design/list.html)
    * [Popover menu](https://docs.nextcloud.com/server/14/developer_manual/design/popovermenu.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/design/html.html)
      * [Progress bar](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#progress-bar)
      * [Checkboxes and radios](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#checkboxes-and-radios)
      * [Buttons](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#buttons)
    * [SCSS](https://docs.nextcloud.com/server/14/developer_manual/design/css.html)
    * [Icons](https://docs.nextcloud.com/server/14/developer_manual/design/icons.html)
  * [Android application development](https://docs.nextcloud.com/server/14/developer_manual/android_library/index.html)
  * [Client APIs](https://docs.nextcloud.com/server/14/developer_manual/client_apis/index.html)
  * [Core development](https://docs.nextcloud.com/server/14/developer_manual/core/index.html)
  * [Bugtracker](https://docs.nextcloud.com/server/14/developer_manual/bugtracker/index.html)
  * [Help and communication](https://docs.nextcloud.com/server/14/developer_manual/commun/index.html)
  * [API Documentation](https://docs.nextcloud.com/server/14/developer_manual/api.html)


[Nextcloud 14 Developer Manual](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/index.html) »
  * [Design guidelines](https://docs.nextcloud.com/server/14/developer_manual/design/index.html) »
  * HTML elements
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/design/html.rst)


* * *
# HTML elements[¶](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#html-elements "Permalink to this headline")
## Progress bar[¶](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#progress-bar "Permalink to this headline")
Nextcloud support and provides an already themed progress bar.
Please use the html5 `progress` element.
![Progress html5](https://docs.nextcloud.com/server/14/developer_manual/_images/progress.png)
```
<progress value="42.79" max="100"></progress>

```

## Checkboxes and radios[¶](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#checkboxes-and-radios "Permalink to this headline")
As default html5 checkboxes & radios are **not** customizable, we created an override using label and `::after` elements.
There are 2 colors:
  * Default themed with the primary color.
  * White colored.


Requirements:
  * You need to have a `label` element **directly** after the `input` element.
  * The input **must** have the `checkbox` or `radio` class.
  * To use the white theme, you **need** to also add the `checkbox--white` or `radio--white` class.
  * Your label **must** have an associated text for accessibility.


![Nextcloud's themed checkboxes](https://docs.nextcloud.com/server/14/developer_manual/_images/checkboxes.png)
```
<input type="checkbox" id="test1" class="checkbox"
       checked="checked">
<label for="test1">Selected</label><br>
<input type="checkbox" id="test2" class="checkbox">
<label for="test2">Unselected</label><br>
<input type="checkbox" id="test3" class="checkbox"
       disabled="disabled">
<label for="test3">Disabled</label><br>
<input type="checkbox" id="test4" class="checkbox">
<label for="test4">Hovered</label><br>

```

![Nextcloud's themed radios](https://docs.nextcloud.com/server/14/developer_manual/_images/radios.png)
```
<input type="radio" id="test1" class="radio"
       checked="checked">
<label for="test1">Selected</label><br>
<input type="radio" id="test2" class="radio">
<label for="test2">Unselected</label><br>
<input type="radio" id="test3" class="radio"
       disabled="disabled">
<label for="test3">Disabled</label><br>
<input type="radio" id="test4" class="radio">
<label for="test4">Hovered</label><br>

```

## Buttons[¶](https://docs.nextcloud.com/server/14/developer_manual/design/html.html#buttons "Permalink to this headline")
[Next ](https://docs.nextcloud.com/server/14/developer_manual/design/css.html "SCSS") [](https://docs.nextcloud.com/server/14/developer_manual/design/popovermenu.html "Popover menu")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
