class User:

    def __init__(self, site=None):
        self.preferences = []
        self.site = site

    def __str__(self):
        return f'Site: {str(self.site)}, Preferences: {str(self.preferences)}'
