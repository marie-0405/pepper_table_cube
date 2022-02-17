from naoqi import ALProxy
class AutonomousLife:
    def __init__(self,ip, port = 9559):
        """ Define parameters

            Parameters
            ----------

            ip : string
               IP value of the robot

        """
        self.port = int(port)
        self.ip = ip
        self.onLoad()

    def onLoad(self):
        try:
            proxy_name ="ALAutonomousLife"
            self.proxy = ALProxy(proxy_name,self.ip,self.port)
            print ( proxy_name + " success")
        except:
            print ( proxy_name + " error")
            return False

        return True
    

    def onRun(self, input_ = "", parameters = {}, parallel = False):
        """ Run action primitive"""
        try:
            self.proxy.setState("input_") # solotary, interactive, safeguard

                
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def onStop(self):
        try:
            if self.proxy.getState != "disabled":
                self.proxy.setState("disabled")
        except:
            print ("Autonomous life:  already disabled")

if __name__ == '__main__':
    auto_life = AutonomousLife(ip=)
    auto_life.onStop()
    