from cmd import Cmd
 
class DBEstPrompt(Cmd):
    prompt = 'dbest> '
    intro = "Welcome to DBEst: a model-based AQP engine! Type ? to list commands"

    def do_exit(self, inp):
        '''exit the application.'''
        print("DBEst closed successfully.")
        return True

    def default(self, inp):
        print(inp)

    def cmdloop(self, intro=None):
        print(self.intro)
        while True:
            try:
                super(DBEstPrompt, self).cmdloop(intro="")
                break
            except KeyboardInterrupt:
                # self.do_exit("")
                print("DBEst closed successfully.")
                return True

    do_EOF = do_exit
 

 
p = DBEstPrompt()
p.cmdloop()