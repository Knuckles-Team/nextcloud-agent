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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html)
    * [1. Migration 1: Schema change](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migration-1-schema-change)
    * [2. Migration 1: Post schema change](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migration-1-post-schema-change)
    * [3. Migration 2: Schema change](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migration-2-schema-change)
    * [Console commands](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#console-commands)
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
  * Migrations
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/migrations.rst)


* * *
# Migrations[¶](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migrations "Permalink to this headline")
In the past, apps had a appinfo/database.xml-file which holds their database schema for installation and update and was a functional method for installing apps which had some trouble with upgrading apps (e.g. apps were not able to rename columns without loosing the data stored in the original column):
```
<?xml version="1.0" encoding="ISO-8859-1" ?>
<database>
         <name>*dbname*</name>
         <create>true</create>
         <overwrite>false</overwrite>
         <charset>utf8</charset>
         <table>
                 <name>*dbprefix*twofactor_backupcodes</name>
                 <declaration>
                       <field>
                               <name>id</name>
                               <type>integer</type>
                               <autoincrement>1</autoincrement>
                               <default>0</default>
                               <notnull>true</notnull>
                               <length>4</length>
                       </field>
 ...

```

The limitations of this method will be bypassed with migrations. A migration can consist of 3 different methods:
  * Pre schema changes
  * Actual schema changes
  * Post schema changes


Apps can have mutiple migrations, which allows a way more flexible updating process. For example, you can rename a column while copying all the content with 3 steps packed in 2 migrations.
After creating migrations for your current database and installation routine, you need to in order to make use of migrations, is to delete the old appinfo/database.xml file. The Nextcloud updater logic only allows to use one or the other. But as soon as the database.xml file is gone, it will look for your migration files in the apps lib/Migration folder.
Note
While in theory you can run any code in the pre- and post-steps, we recommend not to use actual php classes. With migrations you can update from any old version to any new version as long as the migration steps are retained. Since they are also used for installation, you should keep them anyway. But this also means when you change a php class which you use in your migration, the code may be executed on different database/file/code standings when being ran in an upgrade situation.
Note
Since Nextcloud stores, which migrations have been executed already you must not “update” migrations. The recommendation is to keep them untouched as long as possible. You should only adjust it to make sure it still executes, but additional changes to the database should be done in a new migration.
## 1. Migration 1: Schema change[¶](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migration-1-schema-change "Permalink to this headline")
With this step the new column gets created:
```
public function changeSchema(IOutput $output, \Closure $schemaClosure, array $options) {
                   /** @var Schema $schema */
                   $schema = $schemaClosure();

                   $table = $schema->getTable('twofactor_backupcodes');

                   $table->addColumn('user_id', Type::STRING, [
                           'notnull' => true,
                                 'length' => 64,
                   ]);

                   return $schema;
      }

```

## 2. Migration 1: Post schema change[¶](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migration-1-post-schema-change "Permalink to this headline")
In this step the content gets copied from the old to the new column.
Note
This could also be done as part of the second migration as part of a pre schema change
```
public function postSchemaChange(IOutput $output, \Closure $schemaClosure, array $options) {
       $query = $this->db->getQueryBuilder();
       $query->update('twofactor_backupcodes')
               ->set('user_id', 'uid');
       $query->execute();
}

```

## 3. Migration 2: Schema change[¶](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#migration-2-schema-change "Permalink to this headline")
With this the old column gets removed.
```
 public function changeSchema(IOutput $output, \Closure $schemaClosure, array $options) {
        /** @var Schema $schema */
        $schema = $schemaClosure();

        $table = $schema->getTable('twofactor_backupcodes');
        $table->dropColumn('uid');

        return $schema;
}

```

## Console commands[¶](https://docs.nextcloud.com/server/14/developer_manual/app/migrations.html#console-commands "Permalink to this headline")
There are some console commands, which should help developers to create or deal with migrations, which are sometimes only available if you are running your Nextcloud in debug mode:
  * migrations:execute: Executes a single migration version manually.
  * migrations:generate: This is needed to create a new migration file. This takes 2 arguments, first one is the appid, the second one should be the version`of your app as an integer. We recommend to use the major and minor digits of your apps version for that. This allows you to introduce a new migration in your branch for a Nextcloud version if there is already an migration path for a newer one in another branch. Since you can’t change this retroactive, we recommend to leave enough space in between and therefor map the numbers to 3 digits: `1.0.x => 1000, 2.34.x => 2034, etc.
  * migrations:generate-from-schema: Create a migration from the old database.xml.
  * migrations:migrate: Execute a migration to a specified or the latest available version.
  * migrations:status: View the status of a set of migrations.


[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/repair.html "Repair steps") [](https://docs.nextcloud.com/server/14/developer_manual/app/logging.html "Logging")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
