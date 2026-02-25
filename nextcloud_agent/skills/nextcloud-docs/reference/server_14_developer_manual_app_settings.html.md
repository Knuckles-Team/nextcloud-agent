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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html)
    * [Admin](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#admin)
    * [Settings form](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#settings-form)
    * [](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#section)
      * [Personal](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#personal)
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
  * Settings
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/settings.rst)


* * *
# Settings[¶](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#settings "Permalink to this headline")
An app can register both admin settings as well as personal settings.
## Admin[¶](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#admin "Permalink to this headline")
For Nextcloud 10 the admin settings page got reworked. It is not a long list anymore, but divided into sections, where related settings forms are grouped. For example, in the **Sharing** section are only settings (built-in and of apps) related to sharing.
## Settings form[¶](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#settings-form "Permalink to this headline")
For the settings form, three things are necessary:
  1. A class implementing `\OCP\Settings\ISettings`
  2. A template
  3. The implementing class specified in the app’s info.xml


Below is an example for an implementor of the ISettings interface. It is based on the survey_client solution.
```
<?php
namespace OCA\YourAppNamespace\Settings;

use OCA\YourAppNamespace\Collector;
use OCP\AppFramework\Http\TemplateResponse;
use OCP\BackgroundJob\IJobList;
use OCP\IConfig;
use OCP\IDateTimeFormatter;
use OCP\IL10N;
use OCP\Settings\ISettings;

class AdminSettings implements ISettings {

        /** @var Collector */
        private $collector;

        /** @var IConfig */
        private $config;

        /** @var IL10N */
        private $l;

        /** @var IDateTimeFormatter */
        private $dateTimeFormatter;

        /** @var IJobList */
        private $jobList;

        /**
         * Admin constructor.
         *
         * @param Collector $collector
         * @param IConfig $config
         * @param IL10N $l
         * @param IDateTimeFormatter $dateTimeFormatter
         * @param IJobList $jobList
         */
        public function __construct(Collector $collector,
                                                                IConfig $config,
                                                                IL10N $l,
                                                                IDateTimeFormatter $dateTimeFormatter,
                                                                IJobList $jobList
        ) {
                $this->collector = $collector;
                $this->config = $config;
                $this->l = $l;
                $this->dateTimeFormatter = $dateTimeFormatter;
                $this->jobList = $jobList;
        }

        /**
         * @return TemplateResponse
         */
        public function getForm() {

                $lastSentReportTime = (int) $this->config->getAppValue('survey_client', 'last_sent', 0);
                if ($lastSentReportTime === 0) {
                        $lastSentReportDate = $this->l->t('Never');
                } else {
                        $lastSentReportDate = $this->dateTimeFormatter->formatDate($lastSentReportTime);
                }

                $lastReport = $this->config->getAppValue('survey_client', 'last_report', '');
                if ($lastReport !== '') {
                        $lastReport = json_encode(json_decode($lastReport, true), JSON_PRETTY_PRINT);
                }

                $parameters = [
                        'is_enabled' => $this->jobList->has('OCA\Survey_Client\BackgroundJobs\MonthlyReport', null),
                        'last_sent' => $lastSentReportDate,
                        'last_report' => $lastReport,
                        'categories' => $this->collector->getCategories()
                ];

                return new TemplateResponse('yourappid', 'admin', $parameters);
        }

        /**
         * @return string the section ID, e.g. 'sharing'
         */
        public function getSection() {
                return 'survey_client';
        }

        /**
         * @return int whether the form should be rather on the top or bottom of
         * the admin section. The forms are arranged in ascending order of the
         * priority values. It is required to return a value between 0 and 100.
         */
        public function getPriority() {
                return 50;
        }

}

```

The parameters of the constructor will be resolved and an instance created automatically on demand, so that the developer does not need to take care of it.
`getSection` is supposed to return the section ID of the desired admin section. Currently, built-in values are `server`, `sharing`, `encryption`, `logging`, `additional` and `tips-tricks`. Apps can register sections of their own (see below), and also register into sections of other apps.
`getPriority` is used to order forms within a section. The lower the value, the more on top it will appear, and vice versa. The result depends on the priorities of other settings.
Nextcloud will look for the templates in a template folder located in your apps root directory. It should always end on .php, in this case `templates/admin.php` would be the final relative path.
```
<?php
/** @var $l \OCP\IL10N */
/** @var $_ array */

script('myappid', 'admin');         // adds a JavaScript file
style('survey_client', 'admin');    // adds a CSS file
?>

<div id="survey_client" class="section">
        <h2><?php p($l->t('Your app')); ?></h2>

        <p>
                <?php p($l->t('Only administrators are allowed to click the red button')); ?>
        </p>

        <button><?php p($l->t('Click red button')); ?></button>

        <p>
                <input id="your_app_magic" name="your_app_magic"
                           type="checkbox" class="checkbox" value="1" <?php if ($_['is_enabled']): ?> checked="checked"<?php endif; ?> />
                <label for="your_app_magic"><?php p($l->t('Do some magic')); ?></label>
        </p>

        <h3><?php p($l->t('Things to define')); ?></h3>
        <?php
        foreach ($_['categories'] as $category => $data) {
                ?>
                <p>
                        <input id="your_app_<?php p($category); ?>" name="your_app_<?php p($category); ?>"
                                   type="checkbox" class="checkbox your_app_category" value="1" <?php if ($data['enabled']): ?> checked="checked"<?php endif; ?> />
                        <label for="your_app_<?php p($category); ?>"><?php print_unescaped($data['displayName']); ?></label>
                </p>
                <?php
        }
        ?>

        <?php if (!empty($_['last_report'])): ?>

        <h3><?php p($l->t('Last report')); ?></h3>

        <p><textarea title="<?php p($l->t('Last report')); ?>" class="last_report" readonly="readonly"><?php p($_['last_report']);?></textarea></p>

        <em class="last_sent"><?php p($l->t('Sent on: %s', [$_['last_sent']])); ?></em>

        <?php endif; ?>

</div>

```

Then, the implementing class should be added to the info.xml. Settings will be registered upon install and update. When settings are added to an existing, installed and enabled app, it should be made sure that the version is increased so Nextcloud can register the class. It is only possible to register one ISettings implementor.
For a more complex example using embedded templates have a look at the implementation of the **user_ldap** app.
## Section[¶](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#section "Permalink to this headline")
It is also possible that an app registers its own section. This should be done only if there is not fitting corresponding section and the apps settings form takes a lot of screen estate. Otherwise, register to “additional”.
Basically, it works the same way as with the settings form. There are only two differences. First, the interface that must be implemented is `\OCP\Settings\ISection`.
Second, a template is not necessary.
An example implementation of the ISection interface:
```
<?php
namespace OCA\YourAppNamespace\Settings;

use OCP\IL10N;
use OCP\Settings\ISection;

class AdminSection implements ISection {

        /** @var IL10N */
        private $l;

        public function __construct(IL10N $l) {
                $this->l = $l;
        }

        /**
         * returns the ID of the section. It is supposed to be a lower case string
         *
         * @returns string
         */
        public function getID() {
                return 'yourappid'; //or a generic id if feasible
        }

        /**
         * returns the translated name as it should be displayed, e.g. 'LDAP / AD
         * integration'. Use the L10N service to translate it.
         *
         * @return string
         */
        public function getName() {
                return $this->l->t('Translatable Section Name');
        }

        /**
         * @return int whether the form should be rather on the top or bottom of
         * the settings navigation. The sections are arranged in ascending order of
         * the priority values. It is required to return a value between 0 and 99.
         */
        public function getPriority() {
                return 80;
        }

}

```

Also the section must be registered in the app’s info.xml.
### Personal[¶](https://docs.nextcloud.com/server/14/developer_manual/app/settings.html#personal "Permalink to this headline")
Registering personal settings follows and old style yet. Within the app intialisation (e.g. in appinfo/app.php) a method must be called:
```
<?php
\OCP\App::registerPersonal('yourappid', 'personal');

```

Upon opening the personal page, Nextcloud will look for `personal.php` script, execute it and print the output.
[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/logging.html "Logging") [](https://docs.nextcloud.com/server/14/developer_manual/app/backgroundjobs.html "Background jobs \(Cron\)")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
