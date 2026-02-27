[ ![Logo](https://docs.nextcloud.com/server/14/developer_manual/_static/logo-white.png) ](https://docs.nextcloud.com/server/14/developer_manual/index.html)
  * [](https://docs.nextcloud.com/server/14/developer_manual/general/index.html)
    * [Community code of conduct](https://docs.nextcloud.com/server/14/developer_manual/general/code-of-conduct.html)
    * [Development environment](https://docs.nextcloud.com/server/14/developer_manual/general/devenv.html)
    * [Security guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/security.html)
    * [Coding style & general guidelines](https://docs.nextcloud.com/server/14/developer_manual/general/codingguidelines.html)
    * [](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html)
      * [](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#database-performance)
        * [Measuring performance](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#measuring-performance)
      * [Getting help](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#getting-help)
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
  * Performance considerations
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/general/performance.rst)


* * *
# Performance considerations[¶](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#performance-considerations "Permalink to this headline")
This document introduces some common considerations and tips on improving performance of Nextcloud. Speed of Nextcloud is important - nobody likes to wait and often, what is _just slow_ for a small amount of data will become _unusable_ with a large amount of data. Please keep these tips in mind when developing for Nextcloud and consider reviewing your app to make it faster.
## Database performance[¶](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#database-performance "Permalink to this headline")
The database plays an important role in Nextcloud performance. The general rule is: database queries are very bad and should be avoided if possible. The reasons for that are:
  * Roundtrips: Bigger Nextcloud installations have the database not installed on the application server but on a remote dedicated database server. The problem is that database queries then go over the network. These roundtrips can add up significantly if you have a lot of queries.
  * Speed. A lot of people think that databases are fast. This is not always true if you compare it with handling data internally in PHP or in the filesystem or even using key/value based storages. So every developer should always double check if the database is really the best place for the data.
  * Scalability. If you have a big Nextcloud cluster setup you usually have several Nextcloud/Web servers in parallel and a central database and a central storage. This means that everything that happens on the Nextcloud/PHP side can parallelize and can be scaled. Stuff that is happening in the database and in the storage is critical because it only exists once and can’t be scaled so easily.


We can reduce the load on the database by:
  1. Making sure that every query uses an index.
  2. Reducing the overall number of queries.
  3. If you are familiar with cache invalidation you can try caching query results in PHP.


There a several ways to monitor which queries are actually executed on the database.
With MySQL it is very easy with just a bit of configuration:
  1. Slow query log.


If you put this into your my.cnf file, every query that takes longer than one second is logged to a logfile:
```
slow_query_log = 1
slow_query_log_file = /var/log/mysql/mysql-slow.log
long_query_time=1

```

If a query takes more than a second we have a serious problem of course. You can watch it with tail -f /var/log/mysql/mysql-slow.log while using Nextcloud.
  1. log all queries.


If you reduce the long_query_time to zero then every statement is logged. This is super helpful to see what is going on. Just do a tail -f on the logfile and click around in the interface or access the WebDAV interface:
```
slow_query_log = 1
slow_query_log_file = /var/log/mysql/mysql-slow.log
long_query_time=0

```

  1. log queries without an index.


If you increase the long_query_time to 100 and add log-queries-not-using-indexes, all the queries that are not using an index are logged. Every query should always use an index. So ideally there should be no output:
```
log-queries-not-using-indexes
slow_query_log = 1
slow_query_log_file = /var/log/mysql/mysql-slow.log
long_query_time=100

```

### Measuring performance[¶](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#measuring-performance "Permalink to this headline")
If you do bigger changes in the architecture or the database structure you should always double check the positive or negative performance impact. There are a [few nice small scripts](https://github.com/owncloud/administration/tree/master/performance-tests) that can be used for this.
The recommendation is to automatically do 10000 PROPFINDs or file uploads, measure the time and compare the time before and after the change.
## Getting help[¶](https://docs.nextcloud.com/server/14/developer_manual/general/performance.html#getting-help "Permalink to this headline")
If you need help with performance or other issues please ask on our [forums](https://help.nextcloud.com) or on our IRC channel **#nextcloud-dev** on **irc.freenode.net**.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/general/debugging.html "Debugging") [](https://docs.nextcloud.com/server/14/developer_manual/general/codingguidelines.html "Coding style & general guidelines")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
