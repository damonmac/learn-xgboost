fs = require 'fs'
stream = require 'stream'
readline = require 'readline'
async = require 'async'

file = process.env.FILE ? "./pairs.csv"
instream = fs.ReadStream(file)
outstream = new stream
rl = readline.createInterface instream, outstream

rl.on 'line', (aline) ->
  q.push aline
  
q = async.queue (line, done) ->
  data = line.split(",")
  # preserve the first field as the label
  id = data.pop() # remove the pair identifier
  vector = Array(49).fill(0)
  vector[0] = data.shift() # assign the label

  # compare alternating fields
  for t, i in data by 2
    # check for data to compare
    if i < 97 and t.length > 0 and data[i+1].length > 0
      c = data[i+1]
      # console.log "#{i}: #{t} -> #{c}"
      if t is c then vector[(i/2)+1] = 1
      # name parts
      if i < 9
        tnames = t.split(" ")
        cnames = c.split(" ")
        same = 0
        differ = 0
        for tn in tnames
          for cn in cnames
            # console.log tn, cn
            if tn is cn then same++ else differ--
            # console.log "same: #{same} and differ: #{differ}"
        
        # count up the same name parts (optimistic)
        # vector[(i/2)+1] = same

        # count the same name parts, but subtract any differences (balanced)
        vector[(i/2)+1] = if same > 0 then same else differ
      
      # compare years, avoiding 0
      if i is 10 or i is 24 or i is 38 or i is 66
        difference = Math.abs(data[i]-data[i+1])
        vector[(i/2)+1] = if difference < 5 then 5 - difference else 0

  console.log vector.toString().replace(/,/g, " ")
  done()

, 4