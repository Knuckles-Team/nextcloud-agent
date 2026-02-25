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
    * [HTML elements](https://docs.nextcloud.com/server/14/developer_manual/design/html.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/design/css.html)
      * [CSS variables](https://docs.nextcloud.com/server/14/developer_manual/design/css.html#css-variables)
      * [SCSS icon mixins](https://docs.nextcloud.com/server/14/developer_manual/design/css.html#scss-icon-mixins)
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
  * SCSS
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/design/css.rst)


* * *
# SCSS[¶](https://docs.nextcloud.com/server/14/developer_manual/design/css.html#scss "Permalink to this headline")
Since version 12 of Nextcloud, we support SCSS natively. You can migrate your files by simply renaming your `.css` files to `.scss`. The server will automatically compile, cache and serve it. The SCSS file is prioritized. Having two files with the same name and a `scss` & `css` extension will ensure backwards compatibility with <12 versions as scss files will be ignored by the server.
## CSS variables[¶](https://docs.nextcloud.com/server/14/developer_manual/design/css.html#css-variables "Permalink to this headline")
Since Nextcloud 14, app developers should use CSS4 variables so you get the values which Nextcloud defines. This way you can be sure that the theming and accessibility app can dynamically adjust the values.
A list of available variables is listed in the server repository: <https://github.com/nextcloud/server/blob/master/core/css/css-variables.scss>
## SCSS icon mixins[¶](https://docs.nextcloud.com/server/14/developer_manual/design/css.html#scss-icon-mixins "Permalink to this headline")
Since Nextcloud 14, we added some SCSS mixins and functions to add and manage SVG icons.
These functions need to be used to add the icons via background-image. They create a list of every icon used in Nextcloud and create an associated list of variables. This allows us to invert the colors of the SVGs when using the dark theme.
```
/**
* SVG COLOR API
*
* @param string $icon the icon filename
* @param string $dir the icon folder within /core/img if $core or app name
* @param string $color the desired color in hexadecimal
* @param int $version the version of the file
* @param bool [$core] search icon in core
*
* @returns string the url to the svg api endpoint
*/
@mixin icon-color($icon, $dir, $color, $version: 1, $core: false)

// Examples
.icon-menu {
        @include icon-color('menu', 'actions', $color-white, 1, true);
        // --icon-menu: url('/svg/core/actions/menu/ffffff?v=1');
        // background-image: var(--icon-menu)
}
.icon-folder {
        @include icon-color('folder', 'files', $color-black);
        // --icon-folder: url('/svg/files/folder/000000?v=1');
        // background-image: var(--icon-folder)
}

```

More information about the [svg color api](https://docs.nextcloud.com/server/14/developer_manual/design/icons.html#svgcolorapi).
The `icon-black-white` mixin is a shortand for the `icon-color` function but it generates two sets of icons with the suffix `-white` and without (default black).
```
/**
* Create black and white icons
* This will add a default black version of and an additional white version when .icon-white is applied
*/
@mixin icon-black-white($icon, $dir, $version, $core: false)

// Examples
@include icon-black-white('add', 'actions', 1, true);

// Will result in
.icon-add {
        @include icon-color('add', 'actions', $color-black, 1, true);
}
.icon-add-white,
.icon-add.icon-white {
        @include icon-color('add', 'actions', $color-white, 1, true);
}

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/design/icons.html "Icons") [](https://docs.nextcloud.com/server/14/developer_manual/design/html.html "HTML elements")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
