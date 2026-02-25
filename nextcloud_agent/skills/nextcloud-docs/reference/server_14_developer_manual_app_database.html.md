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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/database.html)
    * [Mappers](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#mappers)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#entities)
      * [Types](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#types)
      * [Accessing attributes](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#accessing-attributes)
      * [Custom attribute to database column mapping](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#custom-attribute-to-database-column-mapping)
      * [Slugs](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#slugs)
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
  * Database access
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/database.rst)


* * *
# Database access[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#database-access "Permalink to this headline")
The basic way to run a database query is to use the database connection provided by **OCP\IDBConnection**.
Inside your database layer class you can now start running queries like:
```
<?php
// db/authordao.php

namespace OCA\MyApp\Db;

use OCP\DB\QueryBuilder\IQueryBuilder;
use OCP\IDBConnection;

class AuthorDAO {

    private $db;

    public function __construct(IDBConnection $db) {
        $this->db = $db;
    }

    public function find(int $id) {
        $qb = $this->db->getQueryBuilder();

        $qb->select('*')
           ->from('myapp_authors')
           ->where(
               $qb->expr()->eq('id', $qb->createNamedParameter($id, IQueryBuilder::PARAM_INT))
           );

        $cursor = $qb->execute();
        $row = $cursor->fetch();
        $cursor->closeCursor();

        return $row;
    }

}

```

## Mappers[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#mappers "Permalink to this headline")
The aforementioned example is the most basic way to write a simple database query but the more queries amass, the more code has to be written and the harder it will become to maintain it.
To generalize and simplify the problem, split code into resources and create an **Entity** and a **Mapper** class for it. The mapper class provides a way to run SQL queries and maps the result onto the related entities.
To create a mapper, inherit from the mapper base class and call the parent constructor with the following parameters:
  * Database connection
  * Table name
  * **Optional** : Entity class name, defaults to \OCA\MyApp\Db\Author in the example below


```
<?php
// db/authormapper.php

namespace OCA\MyApp\Db;

use OCP\DB\QueryBuilder\IQueryBuilder;
use OCP\IDBConnection;
use OCP\AppFramework\Db\QBMapper;

class AuthorMapper extends QBMapper {

    public function __construct(IDBConnection $db) {
        parent::__construct($db, 'myapp_authors');
    }


    /**
     * @throws \OCP\AppFramework\Db\DoesNotExistException if not found
     * @throws \OCP\AppFramework\Db\MultipleObjectsReturnedException if more than one result
     */
    public function find(int $id) {
        $qb = $this->db->getQueryBuilder();

        $qb->select('*')
           ->from('myapp_authors')
           ->where(
               $qb->expr()->eq('id', $qb->createNamedParameter($id, IQueryBuilder::PARAM_INT))
           );

        return $this->findEntity($qb);
    }


    public function findAll($limit=null, $offset=null) {
        $qb = $this->db->getQueryBuilder();

        $qb->select('*')
           ->from('myapp_authors')
           ->setMaxResults($limit)
           ->setFirstResult($offset);

        return $this->findEntities($sql);
    }


    public function authorNameCount($name) {
        $qb = $this->db->getQueryBuilder();

        $qb->selectAlias($qb->createFunction('COUNT(*)'), 'count')
           ->from('myapp_authors')
           ->where(
               $qb->expr()->eq('name', $qb->createNamedParameter($name, IQueryBuilder::PARAM_STR))
           );

        $cursor = $qb->execute();
        $row = $cursor->fetch();
        $cursor->closeCursor();

        return $row['count'];
    }

}

```

Note
The cursor is closed automatically for all **INSERT** , **DELETE** , **UPDATE** queries and when calling the methods **findOneQuery** , **findEntities** , **findEntity** , **delete** , **insert** and **update**. For custom calls using execute you should always close the cursor after you are done with the fetching to prevent database lock problems on SQLite
Every mapper also implements default methods for deleting and updating an entity based on its id:
```
$authorMapper->delete($entity);

```

or:
```
$authorMapper->update($entity);

```

## Entities[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#entities "Permalink to this headline")
Entities are data objects that carry all the table’s information for one row. Every Entity has an **id** field by default that is set to the integer type. Table rows are mapped from lower case and underscore separated names to _lowerCamelCase_ attributes:
  * **Table column name** : phone_number
  * **Property name** : phoneNumber


```
<?php
// db/author.php
namespace OCA\MyApp\Db;

use OCP\AppFramework\Db\Entity;

class Author extends Entity {

    protected $stars;
    protected $name;
    protected $phoneNumber;

    public function __construct() {
        // add types in constructor
        $this->addType('stars', 'integer');
    }
}

```

### Types[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#types "Permalink to this headline")
The following properties should be annotated by types, to not only assure that the types are converted correctly for storing them in the database (e.g. PHP casts false to the empty string which fails on PostgreSQL) but also for casting them when they are retrieved from the database.
The following types can be added for a field:
  * integer
  * float
  * boolean


### Accessing attributes[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#accessing-attributes "Permalink to this headline")
Since all attributes should be protected, getters and setters are automatically generated for you:
```
<?php
// db/author.php
namespace OCA\MyApp\Db;

use OCP\AppFramework\Db\Entity;

class Author extends Entity {
    protected $stars;
    protected $name;
    protected $phoneNumber;
}

$author = new Author();
$author->setId(3);
$author->getPhoneNumber()  // null

```

### Custom attribute to database column mapping[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#custom-attribute-to-database-column-mapping "Permalink to this headline")
By default each attribute will be mapped to a database column by a certain convention, e.g. **phoneNumber** will be mapped to the column **phone_number** and vice versa. Sometimes it is needed though to map attributes to different columns because of backwards compatibility. To define a custom mapping, simply override the **columnToProperty** and **propertyToColumn** methods of the entity in question:
```
<?php
// db/author.php
namespace OCA\MyApp\Db;

use OCP\AppFramework\Db\Entity;

class Author extends Entity {
    protected $stars;
    protected $name;
    protected $phoneNumber;

    // map attribute phoneNumber to the database column phonenumber
    public function columnToProperty($column) {
        if ($column === 'phonenumber') {
            return 'phoneNumber';
        } else {
            return parent::columnToProperty($column);
        }
    }

    public function propertyToColumn($property) {
        if ($property === 'phoneNumber') {
            return 'phonenumber';
        } else {
            return parent::propertyToColumn($property);
        }
    }

}

```

### Slugs[¶](https://docs.nextcloud.com/server/14/developer_manual/app/database.html#slugs "Permalink to this headline")
Slugs are used to identify resources in the URL by a string rather than integer id. Since the URL allows only certain values, the entity base class provides a slugify method for it:
```
<?php
$author = new Author();
$author->setName('Some*thing');
$author->slugify('name');  // Some-thing

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/configuration.html "Configuration") [](https://docs.nextcloud.com/server/14/developer_manual/app/schema.html "Database schema")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
