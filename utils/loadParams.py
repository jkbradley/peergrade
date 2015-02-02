
import os, sys

# Load parameters from a text file.
# One parameter per line.  (Blank lines skipped)
# Each line:
# [parameter name] : [parameter value]
# Returns a dictionary mapping parameter names to values.
#
# Notes:
#   Chomps whitespace.
#   Colons ':' are not allowed in names or values.
def loadParams(filepath, VERBOSE = False):
    try:
        fid = open(filepath, 'r')
        line = fid.readline()
        params = {}
        while len(line) > 0:
            i = line.find(':')
            if i < 0:
                line = fid.readline()
                continue
            pname = line[:i].rstrip().lstrip()
            pval = line[(i+1):].rstrip().lstrip()
            params[pname] = pval
            line = fid.readline()
        fid.close()
        if VERBOSE:
            if len(params) == 0:
                print 'loadParams: Loaded NO parameters.\n'
            else:
                print 'loadParams: Loaded parameters:'
                for pname in sorted(params.keys()):
                    print '\t%s\t:\t%s' % (pname, params[pname])
                print '\n'
        return params
    except:
        raise Exception('loadParams failed to read file: %s\n' % filepath)

# Load parameter, and check the type.
# Arguments:
#   params   Parameter dict loaded by loadParams().
#   name     Parameter name.
def getParam(params, name, type_, default=None):
    if name not in params:
        if default is None:
            sys.stderr.write('ERROR: getParam(): name=%s does not exist in params.' \
                             % (name))
            raise Exception('Could not load parameter %s' % name)
        else:
            return type_(default)
    try:
        if (type_ is bool) or (type_ is list):
            val = type_(eval(params[name]))
        else:
            val = type_(params[name])
    except:
        sys.stderr.write('ERROR: getParam(): Parameter %s with value %s could not be converted to type %s' \
                         % (name, str(params[name]), str(type_)))
        raise Exception('Could not load parameter %s' % name)
    return val
