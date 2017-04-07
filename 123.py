import botornot

twitter_app_auth = {
    'consumer_key': '67ZM0Dj6M840f5d5zpl8dPDIx',
    'consumer_secret': 'ONrkWWeAmLvBZNFTqhdRbEoNdmrjRsnIR9BE4tu1l8edCuLpGx',
    'access_token': '838854448407588868-UniQ02TdOKNVkNgWZS7dwqXA6v5v9na',
    'access_token_secret': 'gYoXylBkVvJMPY56BlHjhNFXJXUIp4zIoJCax3h2PwGAN',
    }
bon = botornot.BotOrNot(**twitter_app_auth)

# Check a single account
#result = bon.check_account('@clayadavis')

# Check a sequence of accounts
accounts = ['@jusbieberphotos', '@dtufreak']
results = list(bon.check_accounts_in(accounts))
print results[