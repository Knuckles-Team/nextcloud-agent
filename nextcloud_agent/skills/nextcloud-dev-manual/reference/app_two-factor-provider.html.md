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
  * [](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html)
    * [Implementing a simple two-factor auth provider](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html#implementing-a-simple-two-factor-auth-provider)
    * [Registering a two-factor auth provider](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html#registering-a-two-factor-auth-provider)
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
  * Two-factor providers
  * [ Edit on GitHub](https://github.com/nextcloud/documentation/edit/stable14/developer_manual/app/two-factor-provider.rst)


* * *
# Two-factor providers[¶](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html#two-factor-providers "Permalink to this headline")
Two-factor auth providers apps are used to plug custom second factors into the Nextcloud core. The following code was taken from the [two-factor test app](https://github.com/ChristophWurst/twofactor_test).
## Implementing a simple two-factor auth provider[¶](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html#implementing-a-simple-two-factor-auth-provider "Permalink to this headline")
Two-factor auth providers must implement the `OCP\Authentication\TwoFactorAuth\IProvider` interface. The example below shows a minimalistic example of such a provider.
```
<?php

namespace OCA\TwoFactor_Test\Provider;

use OCP\Authentication\TwoFactorAuth\IProvider;
use OCP\IUser;
use OCP\Template;

class TwoFactorTestProvider implements IProvider {

        /**
         * Get unique identifier of this 2FA provider
         *
         * @return string
         */
        public function getId() {
                return 'test';
        }

        /**
         * Get the display name for selecting the 2FA provider
         *
         * @return string
         */
        public function getDisplayName() {
                return 'Test';
        }

        /**
         * Get the description for selecting the 2FA provider
         *
         * @return string
         */
        public function getDescription() {
                return 'Use a test provider';
        }

        /**
         * Get the template for rending the 2FA provider view
         *
         * @param IUser $user
         * @return Template
         */
        public function getTemplate(IUser $user) {
                // If necessary, this is also the place where you might want
                // to send out a code via e-mail or SMS.

                // 'challenge' is the name of the template
                return new Template('twofactor_test', 'challenge');
        }

        /**
         * Verify the given challenge
         *
         * @param IUser $user
         * @param string $challenge
         */
        public function verifyChallenge(IUser $user, $challenge) {
                if ($challenge === 'passme') {
                        return true;
                }
                return false;
        }

        /**
         * Decides whether 2FA is enabled for the given user
         *
         * @param IUser $user
         * @return boolean
         */
        public function isTwoFactorAuthEnabledForUser(IUser $user) {
                // 2FA is enforced for all users
                return true;
        }

}

```

## Registering a two-factor auth provider[¶](https://docs.nextcloud.com/server/14/developer_manual/app/two-factor-provider.html#registering-a-two-factor-auth-provider "Permalink to this headline")
You need to inform the Nextcloud core that the app provides two-factor auth functionality. Two-factor providers are registered via `info.xml`.
```
<two-factor-providers>
        <provider>OCA\TwoFactor_Test\Provider\TwoFactorTestProvider</provider>
</two-factor-providers>

```

[Next ](https://docs.nextcloud.com/server/14/developer_manual/app/hooks.html "Hooks") [](https://docs.nextcloud.com/server/14/developer_manual/app/users.html "User management")
* * *
© Copyright 2021 Nextcloud GmbH.
Read the Docs v: 14

Versions
    [14](https://docs.nextcloud.com/server/14/developer_manual)     [15](https://docs.nextcloud.com/server/15/developer_manual)     [16](https://docs.nextcloud.com/server/16/developer_manual)     [stable](https://docs.nextcloud.com/server/stable/developer_manual)     [latest](https://docs.nextcloud.com/server/latest/developer_manual)

Downloads


On Read the Docs
     [Project Home](https://docs.nextcloud.com/projects//?fromdocs=)      [Builds](https://docs.nextcloud.com/builds//?fromdocs=)
