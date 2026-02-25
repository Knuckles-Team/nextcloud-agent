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
    * [](https://docs.nextcloud.com/server/14/developer_manual/design/content.html)
      * [Rules and information](https://docs.nextcloud.com/server/14/developer_manual/design/content.html#rules-and-information)
    * [Content list](https://docs.nextcloud.com/server/14/developer_manual/design/list.html)
    * [Popover menu](https://docs.nextcloud.com/server/14/developer_manual/design/popovermenu.html)
    * [HTML elements](https://docs.nextcloud.com/server/14/developer_manual/design/html.html)
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
  * Main content
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/design/content.rst)


* * *
# Main content[¶](https://docs.nextcloud.com/server/14/developer_manual/design/content.html#main-content "Permalink to this headline")
Since 14, we standardized our structure.
Your application will be directly injected into the `#content` div.
```
<header>
        <div class="header-left">
                <!-- apps menu -->
        </div>
        <div class="header-right">
                <!-- search - contactsmenu - settingsmenu - ... -->
        </div>
</header>
<div id="content" class="app-YOURAPPID">
        <div id="app-navigation" class="">
                <div class="app-navigation-new">
                        <!-- app 'new' button -->
                </div>
                <ul id="usergrouplist">
                        <!-- app navigation -->
                </ul>
                <div id="app-settings">
                        <!-- app settings -->
                </div>
        </div>
        <div id="app-content">
                <div id="app-navigation-toggle" class="icon-menu"></div>
                <!-- app-content-wrapper is optional, only use if app-content-list  -->
                <div id="app-content-wrapper">
                        <div class="app-content-list">
                                <!-- app list -->
                        </div>
                        <div class="app-content-details"></div>
                        <!-- app content -->
                </div>
        </div>
        <div id="app-sidebar"></div>
</div>

```

## Rules and information[¶](https://docs.nextcloud.com/server/14/developer_manual/design/content.html#rules-and-information "Permalink to this headline")
  * You cannot nor need to modify the header or the outside elements of your application.
  * The whole body needs to scroll to be compatible with the mobile views. Therefore the sidebar and the app-navigation are fixed/sticky.
  * Unless you application does not require a scrollable area, not not use any overflow properties on the parents of your content.
  * The `app-navigation-toggle` is automatically injected. The navigation hide/show is automatically managed.
  * Do not use `#content-wrapper` anymore
  * If your app is injecting itself by replacing the #content element, make sure to keep the #content id
  * If you use the `app-content-list` standard, the `app-content-details` div will be hidden in mobile mode (full screen). You will need to add the `showdetails` class to the `app-content-list` to show the main content. On mobile view, the whole list/details section (depending on which is shown) will scroll the body


[Next ](https://docs.nextcloud.com/server/14/developer_manual/design/list.html "Content list") [](https://docs.nextcloud.com/server/14/developer_manual/design/navigation.html "Introduction")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
